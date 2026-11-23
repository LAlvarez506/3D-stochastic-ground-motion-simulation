"""
Interfrequency-correlated spectral noise (constrained generation)
=================================================================

Replaces the random amplitude of the windowed white noise, |N(f)|, by a
log-normal factor whose log has the interfrequency correlation of the
normalised residuals (epsilon) of the smoothed effective amplitude spectrum
(EAS) of Bayless & Abrahamson (2019, BSSA 109(3), doi:10.1785/0120180238):

    ln F(f) = sigma(f; Mw) * epsilon(f) ,   epsilon ~ N(0, R)

The phases of the windowed noise are kept, so the time-domain localisation
given by the modulating window is preserved.

Correlation matrices (``correlation_type``)
-------------------------------------------
Bundled in :mod:`stochastic_gm.ba19_data` (239 frequencies, 0.1-24 Hz):

    'within_site'   empirical within-site residual correlation (default).
                    sigma = phi_ss(f, Mw).  The right target when the
                    scenario and the site response are fixed and the noise is
                    the only source of variability.
    'total'         smooth BA19 total-correlation model (tanh form).
                    sigma = total sigma(f, Mw) = sqrt(tau^2 + phi_S2S^2 +
                    phi_ss^2 + c1a^2).  It includes between-event and
                    between-site variability.
    'between_site'  empirical between-site correlation, sigma = phi_S2S(f, Mw).
    'custom'        your own matrix: ``custom=(freqs, matrix, sigma_freqs,
                    sigma)`` (sigma_freqs / sigma may be None if ``sigma`` is
                    given).

The between-event matrix is not available (``'between_event'`` raises).

Standard deviations come from the BA18 EAS model (supplement of BA19) and
depend on magnitude: tau, phi_S2S and phi_ss are given for M <= 4 and M >= 6
and interpolated linearly in between.  ``sigma=`` overrides them with a
constant.

Design
------
* epsilon is generated once per realisation on the 239 BA19 frequencies (the
  Cholesky factor of the matrix is computed once) and linearly interpolated in
  ln f onto the FFT bins, so it does not depend on the record duration.
  Outside 0.1-24 Hz the edge value is held.
* Normalisation (``normalization``):

    'expected'        ln F = sigma*eps - sigma^2 :  E[F^2] = 1 at every
                      frequency for every draw (the rms level of the
                      deterministic spectrum is kept on average; the median of
                      F is exp(-sigma^2) < 1).
    'per_realization' F is divided by sqrt(mean(F^2)) of that draw (Boore
                      style: every realisation has exactly unit mean square).
    'none'            ln F = sigma*eps :  median of F is 1, E[F^2] =
                      exp(2 sigma^2) > 1.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import ba19_data as _D

NORMALIZATIONS = ('expected', 'per_realization', 'none')
APPLY_TO       = ('horizontal', 'all')
CORRELATION_TYPES = ('within_site', 'total', 'between_site', 'custom')


# ---------------------------------------------------------------------------
# BA19 data
# ---------------------------------------------------------------------------

class BA19Data:
    """Matrices and standard deviations bundled in ``ba19_data``."""

    def __init__(self):
        self.freqs = np.asarray(_D.FREQS, dtype=float)
        n = len(self.freqs)
        iu = np.tril_indices(n)
        self._matrices = {}
        for name, tri in _D.LOWER_TRI.items():
            M = np.zeros((n, n))
            M[iu] = np.asarray(tri, dtype=float) / _D.SCALE
            M = M + M.T - np.diag(np.diag(M))
            self._matrices[name] = M
        self._s = {k: np.asarray(v, dtype=float) for k, v in _D.SIGMA.items()}

    def matrix(self, name: str) -> np.ndarray:
        return self._matrices[name].copy()

    def sigma_components(self, Mw: float) -> dict:
        """tau, phi_s2s, phi_ss, c1a and total sigma at the 239 frequencies."""
        w = float(np.clip((Mw - 4.0) / 2.0, 0.0, 1.0))
        s = self._s
        tau = s['s1'] + (s['s2'] - s['s1']) * w
        p2s = s['s3'] + (s['s4'] - s['s3']) * w
        pss = s['s5'] + (s['s6'] - s['s5']) * w
        c1a = s['c1a']
        total = np.sqrt(tau ** 2 + p2s ** 2 + pss ** 2 + c1a ** 2)
        return dict(tau=tau, phi_s2s=p2s, phi_ss=pss, c1a=c1a, total=total)

    def sigma(self, correlation_type: str, Mw: float) -> np.ndarray:
        c = self.sigma_components(Mw)
        return {'within_site': c['phi_ss'], 'between_site': c['phi_s2s'],
                'total': c['total']}[correlation_type]


@functools.lru_cache(maxsize=1)
def load_ba19() -> BA19Data:
    """Bundled BA19 data (built once per process)."""
    return BA19Data()


def load_matrix_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a correlation matrix from a delimited text file for ``custom=``.
    Layout: first row = frequencies (first cell ignored), first column = the
    same frequencies.  Returns (freqs, matrix).
    """
    import csv
    with open(path, newline='') as fh:
        sample = fh.read(2048)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t ')
        rows = [r for r in csv.reader(fh, dialect) if r]
    freqs = np.array([float(x) for x in rows[0][1:]], dtype=float)
    mat = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=float)
    return freqs, mat


def _psd_cholesky(R: np.ndarray, min_eig: float):
    """Cholesky factor of R, repairing it first if it is not positive definite."""
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    info = {'min_eig_raw': float(w[0]), 'repaired': False, 'max_abs_change': 0.0}
    if w[0] < min_eig:
        R2 = (V * np.clip(w, min_eig, None)) @ V.T
        d = np.sqrt(np.diag(R2))
        R2 = R2 / np.outer(d, d)
        info.update(repaired=True, max_abs_change=float(np.max(np.abs(R2 - R))))
        R = R2
    return R, np.linalg.cholesky(R), info


# ---------------------------------------------------------------------------
# Configuration and generator
# ---------------------------------------------------------------------------

@dataclass
class CorrelatedNoiseConfig:
    """
    Options for the correlated spectral noise.

    correlation_type : 'within_site' (default) | 'total' | 'between_site' |
        'custom'  (module docstring).
    normalization : 'expected' | 'per_realization' | 'none'.
    apply_to : 'horizontal' -> only the radial / transverse (horizontal)
        components get the correlated noise, vertical keeps the original
        noise; 'all' -> the same epsilon is used for the vertical too.
    mw : magnitude used for sigma(Mw).  None -> the Mw of the earthquake.
    sigma : constant ln-standard deviation; overrides the BA18 values.
    custom : (freqs, matrix, sigma_freqs, sigma) for correlation_type='custom'.
    min_eig : eigenvalue floor used to repair a non positive-definite matrix.
    """
    correlation_type: str = 'within_site'
    normalization:    str = 'expected'
    apply_to:         str = 'horizontal'
    mw:               Optional[float] = None
    sigma:            Optional[float] = None
    custom:           Optional[tuple] = None
    min_eig:          float = 1e-8

    def __post_init__(self):
        if self.correlation_type == 'between_event':
            raise NotImplementedError(
                "The between-event correlation matrix is not available. "
                "Use 'within_site', 'total', 'between_site' or 'custom'.")
        if self.correlation_type not in CORRELATION_TYPES:
            raise ValueError(f"correlation_type must be one of {CORRELATION_TYPES}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
        if self.apply_to not in APPLY_TO:
            raise ValueError(f"apply_to must be one of {APPLY_TO}")
        if self.correlation_type == 'custom' and self.custom is None:
            raise ValueError("correlation_type='custom' needs custom=(freqs, matrix, "
                             "sigma_freqs, sigma)")

    def build(self, Mw: Optional[float] = None) -> 'CorrelatedNoise':
        """Create the generator.  ``Mw`` is used when ``self.mw`` is None."""
        return CorrelatedNoise(self, Mw)


class CorrelatedNoise:
    """Generator built once from a :class:`CorrelatedNoiseConfig`."""

    def __init__(self, cfg: CorrelatedNoiseConfig, Mw: Optional[float] = None):
        self.cfg = cfg
        self.Mw = cfg.mw if cfg.mw is not None else Mw
        ctype = cfg.correlation_type

        if ctype == 'custom':
            cf, cm, sf, sv = (list(cfg.custom) + [None] * 4)[:4]
            self.nodes = np.asarray(cf, dtype=float)
            R = np.asarray(cm, dtype=float)
            if R.shape != (len(self.nodes),) * 2:
                raise ValueError("custom matrix must be square with len(freqs) rows")
            if np.any(np.diff(self.nodes) <= 0):
                raise ValueError("custom frequencies must be strictly increasing")
            if cfg.sigma is None:
                if sf is None or sv is None:
                    raise ValueError("custom needs sigma_freqs and sigma, or sigma=")
                sig = np.interp(np.log(self.nodes), np.log(np.asarray(sf, float)),
                                np.asarray(sv, float))
        else:
            data = load_ba19()
            self.nodes = data.freqs
            R = data.matrix(ctype)
            sig = None
            if cfg.sigma is None:
                if self.Mw is None:
                    raise ValueError("sigma depends on magnitude: pass Mw to build() "
                                     "or set mw= / sigma= in the config")
                if not 3.0 <= self.Mw <= 8.0:
                    warnings.warn(f"Mw={self.Mw} is outside the BA18 range 3-8; "
                                  "sigma is held at the nearest end value.")
                sig = data.sigma(ctype, self.Mw)

        self.ln_nodes = np.log(self.nodes)
        self.sigma_nodes = (np.full(len(self.nodes), float(cfg.sigma))
                            if cfg.sigma is not None else np.asarray(sig, dtype=float))
        self.R, self.L, self.info = _psd_cholesky(R, cfg.min_eig)
        if self.info['repaired']:
            warnings.warn(
                "Correlation matrix was not positive definite "
                f"(min eigenvalue {self.info['min_eig_raw']:.2e}); eigenvalues "
                f"clipped, max |change| = {self.info['max_abs_change']:.3f}.")

    # ------------------------------------------------------------------
    def epsilon(self, seed: int) -> np.ndarray:
        """Correlated standard-normal vector on the nodes (deterministic in seed)."""
        ss = np.random.SeedSequence([int(seed), 0xBA19])
        z = np.random.default_rng(ss).standard_normal(len(self.nodes))
        return self.L @ z

    def amplitude_factor(self, freqs: np.ndarray, seed: int) -> np.ndarray:
        """Amplitude factor F(f) for positive frequencies ``freqs`` [Hz]."""
        lnf = np.log(np.asarray(freqs, dtype=float))
        # np.interp holds the edge values outside the node range
        eps = np.interp(lnf, self.ln_nodes, self.epsilon(seed))
        sig = np.interp(lnf, self.ln_nodes, self.sigma_nodes)
        ln_f = sig * eps
        if self.cfg.normalization == 'expected':
            ln_f = ln_f - sig ** 2
        F = np.exp(ln_f)
        if self.cfg.normalization == 'per_realization':
            F = F / np.sqrt(np.mean(F ** 2))
        return F

    def apply_to_noise_fft(self, noise_fft: np.ndarray, noise_freqs: np.ndarray,
                           seed: int) -> np.ndarray:
        """
        Return a copy of a windowed-noise FFT (as produced by the simulator,
        ``norm=None``, rms |X| = sqrt(N)) whose positive-frequency amplitudes
        are replaced by sqrt(N) * F(f); the phases are kept.  The negative
        frequencies are set to the complex conjugates; DC and Nyquist are
        left untouched.
        """
        N = len(noise_fft)
        pos = np.flatnonzero(noise_freqs > 0)
        F = self.amplitude_factor(noise_freqs[pos], seed)
        out = noise_fft.copy()
        out[pos] = np.sqrt(N) * F * np.exp(1j * np.angle(noise_fft[pos]))
        out[(N - pos) % N] = np.conj(out[pos])
        return out

    # ------------------------------------------------------------------
    def model_correlation(self, freqs=None) -> np.ndarray:
        """
        Target correlation matrix.  With no argument: on the nodes (exact).
        With ``freqs``: bilinear interpolation in ln f (approximate close to
        the diagonal; pick frequencies from ``self.nodes`` for exact values).
        """
        if freqs is None:
            return self.R.copy()
        from scipy.interpolate import RegularGridInterpolator
        ln = np.clip(np.log(np.asarray(freqs, float)), self.ln_nodes[0], self.ln_nodes[-1])
        interp = RegularGridInterpolator((self.ln_nodes, self.ln_nodes), self.R)
        X, Y = np.meshgrid(ln, ln, indexing='ij')
        R = interp(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(len(ln), len(ln))
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
        return R

    def target_sigma(self, freqs=None) -> np.ndarray:
        """Target ln-standard deviation (on the nodes, or interpolated)."""
        if freqs is None:
            return self.sigma_nodes.copy()
        return np.interp(np.log(np.asarray(freqs, float)), self.ln_nodes, self.sigma_nodes)

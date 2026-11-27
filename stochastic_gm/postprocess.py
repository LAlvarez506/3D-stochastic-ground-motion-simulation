"""
Post-processing: interfrequency-correlated spectral noise on simulated records
================================================================================

The correlated noise of Bayless & Abrahamson (2019, BA19) is applied inside the
simulator for a point source (``SimConfig.corr_noise``).  For a **finite fault**
the motion at the site is the sum of many sub-source records, each built from its
own noise; a correlated epsilon drawn per sub-source would not survive the
summation (the sum of independent correlated draws is a smoothed, de-correlated
residual and the variance is reduced).  The residual has to be imposed on the
*summed* record, which is what this module does.  It works on any set of
simulated (or recorded) three-component accelerograms.

Method (one realization)
------------------------
1.  rFFT of the horizontal components, X_NS(f) and X_EW(f).
2.  Reference ("systematic") spectrum of each component: the Konno-Ohmachi
    smoothed mean-square amplitude, S_c(f) = sqrt(KO[|X_c|^2]).  The smoothing
    bandwidth ``smoothing_b`` (default 10) decides
    what is treated as systematic (path, source, directivity, summation) and what as
    random scatter.  It must be wider than the BA19 smoothing (b = 188.5) that
    defines the epsilon.  The smoothed spectrum is rescaled so that the energy of
    each component is unchanged.  Wider smoothing (smaller b) removes more of the
    record's own scatter and recovers the target sigma and correlation better, but
    also flattens genuine spectral structure; b = 10 is the default, b = 5-40 is
    the useful range (see verification 5).
3.  One correlated epsilon(f) ~ N(0, R) per record (the BA19 matrix chosen in the
    ``CorrelatedNoise`` generator, with sigma(f; Mw) from BA18) and the amplitude
    factor F(f) = exp(sigma*epsilon [- sigma^2]) of the generator (same
    normalization options as in the point source).
4.  mode = 'replace' (default): X_c'(f) = S_c(f) F(f) exp(i angle X_c(f)).
    The fine amplitude scatter of the summed record is replaced by the
    correlated residual, exactly what the point source does with its noise
    (phases are kept, so the time structure of the rupture is kept).
    mode = 'scale': X_c'(f) = X_c(f) F(f).  The record keeps its own scatter
    and the residual is added on top (total variability = own + BA19).
5.  The same F(f) is applied to NS and EW (BA19 define the EAS from both
    horizontals, EAS = sqrt(0.5 (FAS_H1^2 + FAS_H2^2))).  The vertical
    component is only touched with ``apply_vertical=True`` (with its own S_UD).
    Inverse rFFT back to the original length.

With normalization 'expected' (E[F^2] = 1) the expected energy of the record is
unchanged; the median spectrum is lowered by exp(-sigma^2), as in the point source.

Typical use
-----------
>>> from stochastic_gm import CorrelatedNoiseConfig
>>> from stochastic_gm.postprocess import postprocess_eq
>>> new_acc = postprocess_eq(eq, CorrelatedNoiseConfig(correlation_type='within_site'), seed=7)

The input records are never modified.  The seed makes the result reproducible;
record ``i`` uses ``seed + i``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .correlated_noise import CorrelatedNoise, CorrelatedNoiseConfig

MODES = ('replace', 'scale')


# ---------------------------------------------------------------------------
# Konno-Ohmachi smoothing of the mean-square amplitude
# ---------------------------------------------------------------------------

class _KOSmoother:
    """
    Smooth |X|^2 with the Konno-Ohmachi window on a log-spaced evaluation grid
    and interpolate back (in ln f) onto the FFT bins.  Built once per record
    length / dt and reused.
    """

    def __init__(self, freqs: np.ndarray, b: float, n_eval: int = 400):
        self.freqs = freqs
        self.ln_f = np.log(freqs)
        fe = np.geomspace(freqs[0], freqs[-1], n_eval)
        self.ln_fe = np.log(fe)
        x = b * np.log10(freqs[None, :] / fe[:, None])
        with np.errstate(invalid='ignore', divide='ignore'):
            W = (np.sin(x) / x) ** 4
        W[~np.isfinite(W)] = 1.0
        self.W = W / W.sum(axis=1, keepdims=True)

    def __call__(self, power: np.ndarray) -> np.ndarray:
        return np.interp(self.ln_f, self.ln_fe, self.W @ power)


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------

def _process_component(a: np.ndarray, F_full: np.ndarray, pos: np.ndarray,
                       smoother: _KOSmoother, mode: str) -> np.ndarray:
    n = len(a)
    X = np.fft.rfft(a)
    out = X.copy()
    if mode == 'scale':
        out[pos] = X[pos] * F_full
    else:
        P = np.abs(X[pos]) ** 2
        S2 = smoother(P)
        S2 = S2 * (P.sum() / S2.sum())          # smoothing must not change the energy
        S = np.sqrt(S2)
        out[pos] = S * F_full * np.exp(1j * np.angle(X[pos]))
    return np.fft.irfft(out, n)


def postprocess_record(NS: np.ndarray, EW: np.ndarray, UD: np.ndarray, dt: float,
                       noise: CorrelatedNoise, seed: int, mode: str = 'replace',
                       smoothing_b: float = 10., apply_vertical: Optional[bool] = None,
                       _smoother_cache: Optional[dict] = None):
    """
    Impose the correlated residual on one (NS, EW, UD) record.  Returns a new
    (NS, EW, UD) tuple.  ``apply_vertical`` defaults to ``noise.cfg.apply_to == 'all'``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    n = len(NS)
    if len(EW) != n or len(UD) != n:
        raise ValueError("NS, EW and UD must have the same length")
    if apply_vertical is None:
        apply_vertical = noise.cfg.apply_to == 'all'

    freqs = np.fft.rfftfreq(n, dt)
    # positive frequencies without DC; the Nyquist bin is left untouched
    pos = np.arange(1, n // 2 + (1 if n % 2 else 0))
    if len(pos) < 8:
        raise ValueError("record too short for spectral post-processing")
    key = (n, round(dt, 12), smoothing_b)
    cache = _smoother_cache if _smoother_cache is not None else {}
    if key not in cache:
        cache[key] = _KOSmoother(freqs[pos], smoothing_b)
    F = noise.amplitude_factor(freqs[pos], seed)

    new_ns = _process_component(np.asarray(NS, float), F, pos, cache[key], mode)
    new_ew = _process_component(np.asarray(EW, float), F, pos, cache[key], mode)
    if apply_vertical:
        new_ud = _process_component(np.asarray(UD, float), F, pos, cache[key], mode)
    else:
        new_ud = np.asarray(UD, float).copy()
    return new_ns, new_ew, new_ud


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def _as_noise(noise: Union[CorrelatedNoise, CorrelatedNoiseConfig], Mw: Optional[float]):
    if isinstance(noise, CorrelatedNoiseConfig):
        return noise.build(Mw=Mw)
    if isinstance(noise, CorrelatedNoise):
        return noise
    raise TypeError("noise must be a CorrelatedNoiseConfig or a CorrelatedNoise")


def postprocess_records(records: Sequence, dt: float,
                        noise: Union[CorrelatedNoise, CorrelatedNoiseConfig],
                        Mw: Optional[float] = None, seed: int = 0, mode: str = 'replace',
                        smoothing_b: float = 10., apply_vertical: Optional[bool] = None):
    """
    Apply :func:`postprocess_record` to a list of (NS, EW, UD) records (e.g.
    ``eq.sim_acc``).  ``noise`` may be a config (then ``Mw`` is needed unless
    the config fixes ``mw`` or ``sigma``).  Record ``i`` uses seed ``seed + i``.
    """
    gen = _as_noise(noise, Mw)
    cache: dict = {}
    return [postprocess_record(ns, ew, ud, dt, gen, seed + i, mode, smoothing_b,
                               apply_vertical, cache)
            for i, (ns, ew, ud) in enumerate(records)]


def postprocess_eq(eq, noise: Union[CorrelatedNoise, CorrelatedNoiseConfig], seed: int = 0,
                   mode: str = 'replace', smoothing_b: float = 10.,
                   apply_vertical: Optional[bool] = None):
    """
    Post-process the ensemble of a finished :class:`stochastic_gm.EQ` run
    (``eq.sim_acc``), using its time step and magnitude.  Returns the new list;
    ``eq.sim_acc`` is left unchanged.
    """
    if not getattr(eq, 'sim_acc', None):
        raise RuntimeError("eq.sim_acc is empty: run eq.run_simulation() first")
    return postprocess_records(eq.sim_acc, eq.sim_cfg.dt, noise, Mw=eq.Mw, seed=seed,
                               mode=mode, smoothing_b=smoothing_b,
                               apply_vertical=apply_vertical)

# -*- coding: utf-8 -*-
"""
physics.py
----------
Pure numerical functions used by the stochastic ground-motion simulator.

This module contains exactly the functions required by the EQ and PointSource
classes, extracted from the original monolithic Methods.py.  All functions are
stateless (no class, no shared state).

Velocity / distance units
--------------------------
Unless stated otherwise in individual function docstrings:
  - All velocities  : km/s
  - All depths      : km
  - All distances   : km
  - Density         : g/cm³
  - Frequency       : Hz
  - Seismic moment  : dyne·cm

Bug fixes applied vs. the original Methods.py
---------------------------------------------
* site_attenuation : ``low_pass`` NameError fixed to ``lowpass``.
* scale_spectrum   : ``TF is not 'manual'`` changed to ``!=``;
                     the computed ``tf_function`` is now actually applied
                     to ``A_spectrum`` before the spectrum is used.
                     NOTE: the TF parameter here is a **complex** transfer
                     function (amplitude + phase), e.g. from an empirical
                     Green's function approach.  This is different from the
                     amplitude-only GIT TF applied inside
                     PointSource._apply_git_tf.  In the current code this
                     parameter is always None; it is retained for future use.
* ray_propagation_bisection : ``while ... or ...`` corrected to ``and``
                              so the loop exits as soon as convergence is
                              reached instead of always running N iterations.
"""

from __future__ import annotations
import math
import copy
import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.signal
from scipy.optimize import minimize
from scipy.signal import butter
from scipy.spatial import Delaunay
from scipy.signal.windows import tukey



# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def vector_plane_angle(v1: list, v2: list) -> float:
    """Plane angle [radians] between two 2-D vectors."""
    dot  = sum(a * b for a, b in zip(v1, v2))
    norm = math.sqrt(sum(a**2 for a in v1)) * math.sqrt(sum(b**2 for b in v2))
    return math.acos(dot / norm)


def locate_site(reference_global: list, point_global: list,
                reference: list = None, depth: bool = False) -> list:
    """
    Convert geographic coordinates to local Cartesian offsets [km].

    Parameters
    ----------
    reference_global : [lon, lat, z_km]
        Origin of the local coordinate system.
    point_global : [lon, lat, z_km]
        Point to convert.
    reference : [x0_km, y0_km], optional
        Local offset of the origin (default [0., 0.]).
    depth : bool
        If True, carry the depth [km] from point_global into the returned z.

    Returns
    -------
    [x, y, z]  in km.
    """
    if reference is None:
        reference = [0., 0.]
    lon1, lon2 = reference_global[0], point_global[0]
    lat1, lat2 = reference_global[1], point_global[1]
    dx = (lon2 - lon1) * 40000. * math.cos((lat1 + lat2) * math.pi / 360.) / 360.
    dy = (lat2 - lat1) * 40000. / 360.
    z  = point_global[2] if depth else 0.
    return [reference[0] + dx, reference[1] + dy, z]


def rupture_distances(ps, site: list):
    """
    Compute R_rup, R_jb, and R_x from a list of sub-source positions.

    All inputs and outputs in km.

    Parameters
    ----------
    ps : array-like, shape (N, 3) or (3,)
        Sub-source centroids [x, y, z] in km.
    site : list [x, y, z] in km.

    Returns
    -------
    R_rup, R_jb, R_x : float  [km]
    """
    site      = np.array(site,  dtype=float)
    ps_pos    = np.array(ps,    dtype=float)
    scalar    = ps_pos.ndim == 1
    if scalar:
        ps_pos = ps_pos[np.newaxis, :]

    r_rup = np.linalg.norm(ps_pos - site, axis=1)

    ps_surf       = ps_pos.copy()
    ps_surf[:, 2] = 0.
    r_jb          = np.linalg.norm(ps_surf - site, axis=1)

    if scalar:
        R_jb = r_jb[0]
    else:
        hull = Delaunay([(p[0], p[1]) for p in ps_pos])
        R_jb = 0. if hull.find_simplex((site[0], site[1])) >= 0 else r_jb.min()

    return r_rup.min(), R_jb, r_jb.max()


def find_nearest(array, reference) -> int:
    """Index of the element in *array* closest to *reference*."""
    array = np.asarray(array)
    return int(np.abs(array - reference).argmin())


def in_hull(p, hull) -> bool:
    """Return True if point *p* lies inside the convex hull of *hull*."""
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------

def snells_direct_prop(parameters: dict, Solver: bool = False):
    """
    Propagate a direct ray from source to surface using Snell's law.

    All depths and distances in km; velocities in km/s.

    Parameters
    ----------
    parameters : dict
        Keys: ``'z'`` [km], ``'v'`` [km/s], ``'start_point'`` [km],
        ``'target'`` [km], ``'theta'`` [radians].
        ``z`` and ``v`` describe the 1-D layered medium from surface to source.
        ``theta`` is the initial take-off angle [radians].
    Solver : bool
        If True, return the residual (X_ref - X_horizontal) for use by
        the bisection root finder.  If False, return (R [km], theta, t_travel [s]).
    """
    z, v   = parameters['z'], parameters['v']
    pi, pf = parameters['start_point'], parameters['target']
    theta  = parameters['theta']

    X_ref = math.sqrt((pi[0] - pf[0])**2. + (pi[1] - pf[1])**2.)   # km

    z_inv = list(reversed(z))
    v_inv = list(reversed(v))

    R, X, t_travel = 0., 0., 0.
    for i in range(len(v_inv)):
        dz = z_inv[i] - z_inv[i + 1]              # km
        x  = math.tan(theta) * dz                 # km
        r  = math.sqrt(dz**2. + x**2.)            # km
        if i < len(v_inv) - 1:
            theta = math.asin(math.sin(theta) * v_inv[i + 1] / v_inv[i])
        R        += r
        X        += x
        t_travel += r / v_inv[i]                  # km / (km/s) = s

    if Solver:
        return X_ref - X
    return R, theta, t_travel


def ray_propagation_bisection(a: float, b: float, N: int,
                               tolerance: float, data: dict):
    """
    Find the take-off angle that produces a ray arriving at the target via
    bisection on the horizontal-distance residual.

    FIX vs. original: loop condition changed from ``or`` to ``and`` so that
    the iteration stops as soon as the residual drops below *tolerance*
    rather than always running exactly N iterations.

    Parameters
    ----------
    a, b : float
        Initial bracket for the take-off angle [radians].
    N : int
        Maximum number of iterations.
    tolerance : float
        Convergence criterion on the horizontal-distance residual [km].
    data : dict
        Passed directly to ``snells_direct_prop``.

    Returns
    -------
    (R [km], theta [radians], t_travel [s]) or None if no bracket found.
    """
    data_a  = copy.deepcopy(data)
    data_b  = copy.deepcopy(data)
    data_mn = copy.deepcopy(data)

    data_a['theta'] = a
    data_b['theta'] = b
    f_a = snells_direct_prop(data_a, Solver=True)
    f_b = snells_direct_prop(data_b, Solver=True)

    if f_a * f_b >= 0:
        print("Ray propagation – bisection method fails (bracket does not straddle root).")
        print(data)
        return None

    a_n, b_n  = a, b
    error     = 1e9
    counter   = 0

    # FIX: use `and` so the loop exits on convergence, not just after N steps.
    while error > tolerance and counter < N:
        mn = (a_n + b_n) / 2.
        data_mn['theta'] = mn
        f_mn = snells_direct_prop(data_mn, Solver=True)

        data_a['theta'] = a_n
        f_an = snells_direct_prop(data_a, Solver=True)

        if f_an * f_mn < 0.:
            b_n = mn
        else:
            a_n = mn

        error   = abs(f_mn)
        counter += 1

    if error <= tolerance:
        return snells_direct_prop(data_mn)
    else:
        print(f"Ray propagation – no solution found after {N} iterations.")
        return None


def FS_pwaves(theta: float, vp: float, vs: float):
    """
    Free-surface amplification factors for P-waves (vertical and radial).

    The formula depends only on the velocity *ratio* vp/vs, so any consistent
    unit (km/s or m/s) may be used as long as both vp and vs are in the same unit.

    Parameters
    ----------
    theta : float
        Incidence angle [radians].
    vp, vs : float
        P- and S-wave velocities at the surface [km/s].

    Returns
    -------
    (P_vertical, P_radial)
    """
    j_p = math.asin(math.sin(theta) / vp * vs)
    p   = math.sin(j_p) / vs
    X   = 1. / vs**2 - 2. * p**2
    Y   = 4. * math.cos(theta) * math.cos(j_p) / (vp * vs)
    Z   = X**2 + p**2 * Y
    return abs(2. * X / (vs**2 * Z)), Y / (vs**2 * Z)


def FS_swaves(theta: float, vp: float, vs: float):
    """
    Free-surface amplification factors for SV-waves (vertical and radial).

    The formula depends only on the velocity *ratio* vs/vp, so any consistent
    unit (km/s or m/s) may be used as long as both vp and vs are in the same unit.

    Parameters
    ----------
    theta : float
        Incidence angle [radians].
    vp, vs : float
        P- and S-wave velocities [km/s].

    Returns
    -------
    (SV_vertical, SV_radial)
    """
    C = vs / vp
    if theta <= math.asin(C):
        alpha     = math.asin(vp / vs * math.sin(theta))
        denom     = (math.cos(2. * theta)**2.
                     + C**2. * math.sin(2. * theta) * math.sin(2. * alpha))
        f1        = -(math.cos(2. * theta)**2.
                      - C**2. * math.sin(2. * theta) * math.sin(2. * alpha)) / denom
        f2        = 2. * C * math.sin(2. * theta) * math.cos(2. * theta) / denom
        SV_radial   = 1. - f1 + f2 * math.sin(alpha) / math.cos(theta)
        SV_vertical = 1. + f1 + f2 * math.cos(alpha) / math.sin(theta)
    else:
        a     = math.cos(2. * theta)**2. * math.cos(theta)
        b     = math.sqrt(math.sin(theta)**2. - C**2.) * math.sin(2. * theta)**2.
        R     = math.sqrt(a**2. + b**2.)
        denom = (math.cos(2. * theta)**4.
                 + 4. * (math.sin(theta)**2. - C**2.)
                 * math.sin(2. * theta)**2. * math.sin(theta)**2.)
        SV_radial   = abs(2. * math.cos(2. * theta) / denom * R / math.cos(theta))

        a   = 2. * math.sqrt(math.sin(theta)**2. - C**2.) * math.sin(2. * theta) * math.sin(theta)
        b   = math.cos(2. * theta)**2.
        R   = math.sqrt(a**2. + b**2.)
        num = 2. * math.sqrt(math.sin(theta)**2. - C**2.) * math.sin(2. * theta)
        SV_vertical = num / denom * R / math.sin(theta)

    return SV_vertical, SV_radial


def density_profile(betha: np.ndarray) -> np.ndarray:
    """
    Generic density model (Boore, 2003).

    Parameters
    ----------
    betha : array-like
        S-wave velocity [km/s].  **Must be in km/s.**
        Passing velocities in m/s will produce physically unrealistic
        densities (e.g. ~255 g/cm³ instead of ~2.7 g/cm³).

    Returns
    -------
    rho : ndarray, same shape as betha  [g/cm³]
    """
    betha = np.asarray(betha, dtype=float)
    return 2.5 + (betha - 0.3) * (0.3 / 3.2)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def white_noise(n_samples: int, standard_dev: float = 1.,
                seed: int = None) -> np.ndarray:
    """
    Generate zero-mean Gaussian white noise.

    Parameters
    ----------
    n_samples : int
    standard_dev : float
    seed : int, optional

    Returns
    -------
    noise : ndarray, shape (n_samples,)
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    rng = np.random.default_rng(seed)
    return rng.normal(0., standard_dev, size=n_samples)


def FFT(signal: np.ndarray, dt: float, norm=None):
    """
    Compute the FFT of a real signal.

    Returns
    -------
    (fft, freqs)
    """
    N     = len(signal)
    freqs = np.fft.fftfreq(N, d=dt)
    fft   = np.fft.fft(signal, norm=norm)
    return fft, freqs


def IFFT(f: np.ndarray, norm=None) -> np.ndarray:
    """Inverse FFT."""
    return np.fft.ifft(f, norm=norm)


def norm_power_spec(noise: np.ndarray, dt: float) -> float:
    """
    Normalisation factor so that the windowed noise has unit mean power
    spectral amplitude after the FFT.
    """
    N     = len(noise)
    freqs = np.fft.fftfreq(N, d=dt)
    fft   = np.fft.fft(noise, norm=None)
    mask  = freqs > 0
    fas   = 2. / N * np.abs(fft[mask])
    return float((1. / np.mean(fas**2))**0.5)


def scale_spectrum(A_spectrum: np.ndarray,
                   noise_fft:   np.ndarray,
                   noise_freqs: np.ndarray,
                   time:        np.ndarray,
                   standard_freqs: np.ndarray,
                   factor:      float  = None,
                   TF:          object = None,
                   incidence:   float  = None,
                   polarization: str   = None,
                   wave_type:   str    = None,
                   norm:        object = None):
    """
    Convolve a deterministic amplitude spectrum with a windowed noise FFT
    and return the corresponding time-domain signal.

    NOTE on the TF parameter
    ------------------------
    ``TF``, when provided, is a **complex** transfer function (amplitude AND
    phase), intended for cases where the full empirical propagation response
    is known (e.g., an empirical Green's function approach that retains phase
    information from observed records).  This is **different** from the
    amplitude-only GIT transfer functions applied at spectrum construction time
    inside PointSource._apply_git_tf, which modify only the spectral shape
    without phase information.
    In the current implementation TF is always None; the parameter is retained
    for future use.

    FIX vs. original: the transfer function ``TF``, when provided, is now
    actually applied to ``A_spectrum``.  Previously ``tf_function`` was
    computed but never used.

    Parameters
    ----------
    A_spectrum : ndarray
        Deterministic FAS defined on *standard_freqs*.
    noise_fft, noise_freqs : ndarray
        FFT and frequency axis of the windowed white noise.
    time : ndarray
        Time axis corresponding to the noise segment.
    standard_freqs : ndarray
        Frequency axis for *A_spectrum*.
    factor : float, optional
        Additional amplitude scaling factor.
    TF : dict or 'manual', optional
        Complex transfer function keyed by incidence angle (degrees).
    incidence : float, optional
        Incidence angle [radians], used to look up TF.
    polarization, wave_type : str, optional
        Keys used to look up the correct TF component.
    norm : optional
        Passed to ``np.fft.ifft``.

    Returns
    -------
    (sim_signal, time_array)
    """
    A_spectrum = np.array(A_spectrum, dtype=float)

    # --- Apply complex transfer function if provided -----------------------
    # FIX: tf_function is now evaluated and multiplied into A_spectrum.
    if TF is not None and TF != 'manual':
        if incidence is not None:
            angles      = np.array(list(TF.keys()))
            angles_rad  = angles * np.pi / 180.
            idx         = int(np.argmin(np.abs(angles_rad - incidence)))
            tf_freqs    = TF[angles[idx]][wave_type][polarization]['freqs']
            tf_vals     = TF[angles[idx]][wave_type][polarization]['comp']
            tf_function = scipy.interpolate.interp1d(
                tf_freqs, tf_vals, kind='linear',
                bounds_error=False,
                fill_value=(tf_vals[0], tf_vals[-1])
            )
            A_spectrum = A_spectrum * tf_function(standard_freqs)
        else:
            print("scale_spectrum: incidence angle required when TF is provided.")

    # --- Build spectrum function on the noise frequency grid ----------------
    spectrum_fn = scipy.interpolate.interp1d(
        standard_freqs, A_spectrum / 2.,   # /2: half energy to negative branch
        kind='linear', bounds_error=False,
        fill_value=(A_spectrum[0] / 2., A_spectrum[-1] / 2.)
    )

    N         = len(noise_fft)
    mask      = noise_freqs > 0
    fas_noise = np.abs(noise_fft[mask])
    phases    = np.angle(noise_fft[mask])

    scaled_fas = spectrum_fn(noise_freqs[mask]) * fas_noise
    fft_pos    = scaled_fas * np.exp(1j * phases)
    fft_neg    = np.conj(fft_pos)

    if N % 2 == 0:
        DC      = np.array([noise_fft[0] * spectrum_fn(standard_freqs[0])])
        Nyquist = np.array([noise_fft[N // 2] * spectrum_fn(abs(noise_freqs[N // 2]))])
        A       = np.concatenate((DC, fft_pos, Nyquist, np.flip(fft_neg)))
    else:
        DC = np.array([noise_fft[0] * spectrum_fn(standard_freqs[0])])
        A  = np.concatenate((DC, fft_pos, np.flip(fft_neg)))

    sim_signal = IFFT(A, norm=norm).real * (factor if factor is not None else 1.)
    time_array = np.linspace(0., max(time), len(sim_signal))
    return sim_signal, time_array


# ---------------------------------------------------------------------------
# Site functions
# ---------------------------------------------------------------------------
def site_amplification(freq, amp_freq, amp):
    if (amp_freq is None or amp is None or 
        np.ndim(amp) == 0 or np.ndim(amp_freq) == 0):
        return np.ones_like(np.array(freq))

    amp_freq = np.array(amp_freq)
    amp      = np.array(amp)
    freq     = np.array(freq)

    fn = scipy.interpolate.interp1d(
        amp_freq, amp,
        kind='linear',
        bounds_error=False,
        fill_value=(amp[0], amp[-1])
    )
    return fn(freq)

def site_attenuation(freq:      np.ndarray,
                     kappa:     float = None,
                     f_max:     float = None,
                     lowpass:   dict  = None,
                     f_max_exp: float = 8.,
                     fk:        float = None) -> np.ndarray:
    """
    Compute the high-frequency site attenuation filter D(f).

    Exactly one of *kappa*, *f_max*, or *lowpass* should be provided.

    FIX vs. original: ``low_pass`` NameError corrected to ``lowpass``.

    Parameters
    ----------
    freq : array-like
        Frequency axis [Hz].
    kappa : float, optional
        Anderson–Hough kappa [s].
    f_max : float, optional
        f_max diminution parameter [Hz].
    lowpass : dict, optional
        Butterworth low-pass with keys ``'fmax'`` and ``'order'``.
    f_max_exp : float
        Exponent in the f_max filter.
    fk : float, optional
        Frequency [Hz] at which kappa attenuation begins.
        NOTE: if a ValueError is raised about providing exactly one of
        kappa / f_max / lowpass, verify that SiteConfig.fk is either a
        positive float or the string 'corner', not an unexpected type.

    Returns
    -------
    D : ndarray
    """
    freq = np.asarray(freq, dtype=float)

    if f_max is not None and lowpass is None and kappa is None:
        D = (1. + (freq / f_max) ** f_max_exp) ** -0.5

    elif lowpass is not None and f_max is None and kappa is None:
        f_Max  = lowpass['fmax']
        order  = lowpass['order']
        b, a   = butter(order, f_Max, 'low', analog=True)
        w, h   = scipy.signal.freqs(b, a)
        D      = np.interp(freq, w, np.abs(h))

    elif kappa is not None and f_max is None and lowpass is None:
        if fk is not None:
            # Attenuation begins only above fk [Hz]
            D = np.where(freq <= fk,
                         np.ones_like(freq),
                         np.exp(-math.pi * kappa * (freq - fk)))
        else:
            D = np.exp(-math.pi * kappa * freq)

    else:
        raise ValueError(
            "site_attenuation: exactly one of kappa, f_max, or lowpass must be provided. "
            "If SiteConfig.fk is set unexpectedly, verify it is a positive float or "
            "the string 'corner' (see SiteConfig docstring)."
        )

    return D


# ---------------------------------------------------------------------------
# Window / duration
# ---------------------------------------------------------------------------

def window_function(window_params: dict, Tgm: float, dt: float,
                    decay_value: float = 0.005):
    """
    Saragoni-Hart modulating window.

    Parameters
    ----------
    window_params : dict
        Keys: ``'f_tgm'``, ``'epsilon_window'``, ``'nu_window'``.
    Tgm : float
        Total ground-motion duration [s].
    dt : float
        Time step [s].
    decay_value : float
        Amplitude at which the window is considered to have decayed.

    Returns
    -------
    (time, w) : (ndarray, ndarray)
        Time axis [s] and window amplitudes.
    """
    Tw  = window_params['f_tgm'] * Tgm
    eps = window_params['epsilon_window']
    nu  = window_params['nu_window']
    b   = -(eps * math.log(nu)) / (1. + eps * (math.log(eps) - 1.))
    c   = b / eps
    a   = (math.e / eps) ** b

    def residual(ti):
        return (a * (ti / Tw) ** b * math.exp(-c * ti / Tw) - decay_value) ** 2.

    res   = minimize(residual, Tgm, method='L-BFGS-B', tol=1e-3,
                     bounds=((eps * Tw, 1000.),))
    t_end = float(res.x[0])
    n     = int(t_end / dt)
    time  = np.linspace(0., t_end, n)
    w     = a * (time / Tw) ** b * np.exp(-c * time / Tw)
    return time, w
# ============================================================
# RESPONSE SPECTRUM (Newmark-β, 5% damping)
# ============================================================
# def response_spectrum(acc, dt, periods, damping=0.05):
#     """
#     Pseudo-spectral acceleration (PSA) via Newmark average-acceleration.
    
#     Parameters
#     ----------
#     acc     : array-like, ground acceleration time series [any unit, e.g. cm/s²]
#     dt      : float, time step [s]
#     periods : array-like, oscillator periods [s]
#     damping : float, fraction of critical damping (default 0.05 = 5%)

#     Returns
#     -------
#     SA : np.ndarray, pseudo-spectral acceleration [same units as acc]
#     """
#     beta_n, gamma = 0.25, 0.5

#     periods = np.atleast_1d(np.array(periods, dtype=np.float64))
#     periods = periods[periods > 1e-6]
#     acc     = np.asarray(acc, dtype=np.float64)
#     n       = len(acc)
#     SA      = np.zeros(len(periods))

#     for k, T in enumerate(periods):
#         w  = 2.0 * np.pi / T
#         c  = 2.0 * damping * w    # mass-normalised damping coefficient [s⁻¹]
#         ks = w ** 2                # mass-normalised stiffness [s⁻²]

#         # Effective stiffness (constant for linear system)
#         A = ks + gamma / (beta_n * dt) * c + 1.0 / (beta_n * dt**2)

#         u = np.zeros(n)
#         v = np.zeros(n)
#         a = np.zeros(n)
#         a[0] = -acc[0]            # initial relative acceleration

#         for i in range(n - 1):
#             # Right-hand side — Newmark absolute step
#             rhs = (
#                 -acc[i + 1]                                    # excitation
#                 + u[i] / (beta_n * dt**2)
#                 + v[i] / (beta_n * dt)
#                 + (1.0 / (2.0 * beta_n) - 1.0) * a[i]
#                 + c * (
#                     gamma / (beta_n * dt) * u[i]
#                     + (gamma / beta_n - 1.0) * v[i]
#                     + dt * (gamma / (2.0 * beta_n) - 1.0) * a[i]
#                 )
#             )

#             u[i + 1] = rhs / A

#             a[i + 1] = (
#                 (u[i + 1] - u[i]) / (beta_n * dt**2)
#                 - v[i] / (beta_n * dt)
#                 - (1.0 / (2.0 * beta_n) - 1.0) * a[i]
#             )

#             v[i + 1] = v[i] + dt * (1.0 - gamma) * a[i] + dt * gamma * a[i + 1]

#         # Pseudo-spectral acceleration
#         SA[k] = np.max(np.abs(u)) * ks

#     return SA

from numba import njit

@njit(cache=True)
def _newmark_sdof(acc, dt, w, c, ks, beta_n, gamma):
    n  = len(acc)
    A  = ks + gamma / (beta_n * dt) * c + 1.0 / (beta_n * dt**2)
    u  = np.zeros(n)
    v  = np.zeros(n)
    a  = np.zeros(n)
    a[0] = -acc[0]
    for i in range(n - 1):
        rhs = (
            -acc[i + 1]
            + u[i] / (beta_n * dt**2)
            + v[i] / (beta_n * dt)
            + (1.0 / (2.0 * beta_n) - 1.0) * a[i]
            + c * (
                gamma / (beta_n * dt) * u[i]
                + (gamma / beta_n - 1.0) * v[i]
                + dt * (gamma / (2.0 * beta_n) - 1.0) * a[i]
            )
        )
        u[i + 1] = rhs / A
        a[i + 1] = (
            (u[i + 1] - u[i]) / (beta_n * dt**2)
            - v[i] / (beta_n * dt)
            - (1.0 / (2.0 * beta_n) - 1.0) * a[i]
        )
        v[i + 1] = v[i] + dt * (1.0 - gamma) * a[i] + dt * gamma * a[i + 1]
    return np.max(np.abs(u)) * ks


def response_spectrum(acc, dt, periods, damping=0.05):
    """
    Pseudo-spectral acceleration (PSA) via Newmark average-acceleration
    (β=1/4, γ=1/2), JIT-compiled with numba for speed.

    Parameters
    ----------
    acc     : array-like
        Ground acceleration time series [any unit, e.g. cm/s²].
    dt      : float
        Time step [s].
    periods : array-like
        Oscillator natural periods [s]. Values ≤ 1×10⁻⁶ s are ignored.
    damping : float, optional
        Fraction of critical damping (default 0.05 = 5 %).

    Returns
    -------
    SA : np.ndarray, shape (len(periods),)
        Pseudo-spectral acceleration [same units as acc].
        SA[k] = ω_k² × max|u_rel(t)| for the k-th period.
    """
    beta_n, gamma = 0.25, 0.5
    periods = np.atleast_1d(np.array(periods, dtype=np.float64))
    periods = periods[periods > 1e-6]
    acc     = np.asarray(acc, dtype=np.float64)
    SA      = np.zeros(len(periods))
    for k, T in enumerate(periods):
        w  = 2.0 * np.pi / T
        c  = 2.0 * damping * w
        ks = w ** 2
        SA[k] = _newmark_sdof(acc, dt, w, c, ks, beta_n, gamma)
    return SA
# --------------------------------------------------------------------------- #
# Helper: RotD50 of two horizontal components
# --------------------------------------------------------------------------- #
def rotdpp(comp1, comp2, dt, periods, pp):
    """
    Compute RotD50 response spectrum from two horizontal components.
    Rotates through 180 azimuths (1° step) and returns the median
    (50th percentile) spectral acceleration at each period.

    Parameters
    ----------
    comp1, comp2 : 1-D array  – horizontal acceleration time-series
    dt           : float      – time step [s]
    periods      : array-like – oscillator periods [s]
    pp           : float - Percentile of the cmbination pool (50 for RotD50)

    Returns
    -------
    sa_rotd50 : ndarray, shape (len(periods),)
    """
    angles = np.deg2rad(np.arange(0, 180, 1))          # 180 azimuths
    sa_all = np.zeros((len(angles), len(periods)))

    for k, theta in enumerate(angles):
        rotated = np.cos(theta) * comp1 + np.sin(theta) * comp2
        sa_all[k] = response_spectrum(rotated, dt, periods)

    return np.percentile(sa_all, pp, axis=0)
# --------------------------------------------------------------------------- #
# Helper: Generic signal process
# --------------------------------------------------------------------------- #
def preprocess_acc(sig, dt, taper_fraction=0.05):
    """
    Remove mean and apply a Tukey (split-cosine) taper before spectral analysis.
    Prevents Newmark-β overflow from DC offset or edge discontinuities.

    Parameters
    ----------
    sig             : 1-D ndarray  – acceleration time-series
    dt              : float        – time step [s]
    taper_fraction  : float        – fraction of signal tapered at each end (default 5 %)

    Returns
    -------
    sig_clean : 1-D ndarray
    """
    sig = sig - np.mean(sig)                        # remove DC offset
    sig = sig * tukey(len(sig), alpha=taper_fraction)  # taper edges
    return sig

# --------------------------------------------------------------------------- #
# Helper: Geo distance estimator
# --------------------------------------------------------------------------- #
def geo_distance_km(point_a, point_b):
    """
    Hypocentral distance in km between two points given in
    geographic coordinates [lon_deg, lat_deg, depth_km].
    Uses a flat-Earth approximation (valid for distances < ~200 km).

    Parameters
    ----------
    point_a, point_b : [lon_deg, lat_deg, depth_km]

    Returns
    -------
    distance : float [km]
    """
    lat_mean = np.deg2rad((point_a[1] + point_b[1]) / 2.0)

    km_per_deg_lat = 111.132                          # ~constant
    km_per_deg_lon = 111.132 * np.cos(lat_mean)      # shrinks toward poles

    dx = (point_a[0] - point_b[0]) * km_per_deg_lon  # E–W  [km]
    dy = (point_a[1] - point_b[1]) * km_per_deg_lat  # N–S  [km]
    dz =  point_a[2] - point_b[2]                    # depth [km]

    return np.sqrt(dx**2 + dy**2 + dz**2)

# --------------------------------------------------------------------------- #
# Helper: Afshari_Stewart 2016 GMPE for ground motion duration
# --------------------------------------------------------------------------- #
def afshari_stewart_2016(Mw: float, Rrup: float, rake: float,
                          vs30: float = 760., z1pt0: float = None,
                          imt: str = 'rsd595') -> tuple:
    """
    Afshari & Stewart (2016) significant duration GMPE.
    Active shallow crustal regions.

    Parameters
    ----------
    Mw    : moment magnitude
    Rrup  : closest distance to rupture [km]
    rake  : rake angle [deg] — controls style-of-faulting term
    vs30  : time-averaged shear-wave velocity in top 30 m [m/s]
            default 760 m/s (reference rock)
    z1pt0 : depth to Vs=1.0 km/s [m]; if None → estimated from Vs30
            using the California model of Chiou & Youngs (2014)
    imt   : 'rsd575', 'rsd595', or 'rsd2080'

    Returns
    -------
    median  : median significant duration [s]
    sigma   : total standard deviation [log units]
    tau     : inter-event std dev [log units]
    phi     : intra-event std dev [log units]
    """
    import math

    # ------------------------------------------------------------------
    # Coefficients — Table 2 of Afshari & Stewart (2016)
    # ------------------------------------------------------------------
    COEFFS = {
        'rsd575': dict(
            m1=5.35, m2=7.15,
            b0R=0.7806, b0N=1.555,  b0SS=1.279,
            b1R=7.061,  b1N=4.992,  b1SS=5.578,
            b2=0.9011,  b3=-1.684,
            c1=0.1159,  c2=0.1065,  c3=0.0682,
            c4=-0.2246, c5=0.0006,  vref=368.2,
            tau1=0.28, tau2=0.25, phi1=0.54, phi2=0.41,
        ),
        'rsd595': dict(
            m1=5.20, m2=7.40,
            b0R=1.612,  b0N=2.541,  b0SS=2.302,
            b1R=4.536,  b1N=3.170,  b1SS=3.467,
            b2=0.9443,  b3=-3.911,
            c1=0.3165,  c2=0.2539,  c3=0.0932,
            c4=-0.3183, c5=0.0006,  vref=369.9,
            tau1=0.25, tau2=0.19, phi1=0.43, phi2=0.35,
        ),
        'rsd2080': dict(
            m1=5.20, m2=7.40,
            b0R=0.7729, b0N=1.409,  b0SS=0.8804,
            b1R=6.579,  b1N=4.778,  b1SS=6.188,
            b2=0.7414,  b3=-3.164,
            c1=0.0646,  c2=0.0865,  c3=0.0373,
            c4=-0.4237, c5=0.0005,  vref=369.6,
            tau1=0.30, tau2=0.19, phi1=0.56, phi2=0.45,
        ),
    }
    CONSTANTS = {'mstar': 6.0, 'r1': 10.0, 'r2': 50.0,
                 'v1': 600.0, 'dz1ref': 200.0}

    C = COEFFS[imt.lower()]

    # ------------------------------------------------------------------
    # Style-of-faulting term
    # ------------------------------------------------------------------
    if 45. <= rake <= 135.:       # reverse
        b0, b1 = C['b0R'], C['b1R']
    elif -135. <= rake <= -45.:   # normal
        b0, b1 = C['b0N'], C['b1N']
    else:                         # strike-slip
        b0, b1 = C['b0SS'], C['b1SS']

    # ------------------------------------------------------------------
    # Source term (equation 3–6)
    # ------------------------------------------------------------------
    if Mw <= C['m1']:
        T_source = b0
    else:
        M0 = 10. ** (1.5 * Mw + 16.05)        # dyne·cm

        if Mw > C['m2']:
            ln_sd = b1 + C['b2'] * (C['m2'] - CONSTANTS['mstar']) \
                       + C['b3'] * (Mw - C['m2'])
        else:
            ln_sd = b1 + C['b2'] * (Mw - CONSTANTS['mstar'])

        stress_drop = math.exp(ln_sd)          # bar
        f0 = 4.9e6 * 3.2 * (stress_drop / M0) ** (1. / 3.)   # Hz
        T_source = 1. / f0                     # s

    # ------------------------------------------------------------------
    # Path term (equation 7)
    # ------------------------------------------------------------------
    if Rrup <= CONSTANTS['r1']:
        f_p = C['c1'] * Rrup
    elif Rrup <= CONSTANTS['r2']:
        f_p = (C['c1'] * CONSTANTS['r1']
               + C['c2'] * (Rrup - CONSTANTS['r1']))
    else:
        f_p = (C['c1'] * CONSTANTS['r1']
               + C['c2'] * (CONSTANTS['r2'] - CONSTANTS['r1'])
               + C['c3'] * (Rrup - CONSTANTS['r2']))

    # ------------------------------------------------------------------
    # Site term (equations 9–11)
    # ------------------------------------------------------------------
    # z1pt0 from California Vs30 model if not provided (equation 11)
    if z1pt0 is None:
        ln_z1 = ((-7.15 / 4.) *
                 math.log((vs30**4. + 570.94**4.) /
                          (1360.**4. + 570.94**4.)) - math.log(1000.))
        z1pt0 = math.exp(ln_z1)   # m

    dz1 = z1pt0 - math.exp(
        (-7.15 / 4.) * math.log((vs30**4. + 570.94**4.) /
                                 (1360.**4. + 570.94**4.))
        - math.log(1000.)
    )
    # Basin term (equation 9)
    f_dz1 = C['c5'] * min(dz1, CONSTANTS['dz1ref'])

    # Vs30 term
    if vs30 > CONSTANTS['v1']:
        f_vs30 = C['c4'] * math.log(CONSTANTS['v1'] / C['vref'])
    else:
        f_vs30 = C['c4'] * math.log(vs30 / C['vref'])

    f_s = f_vs30 + f_dz1

    # ------------------------------------------------------------------
    # Total ln(duration)
    # ------------------------------------------------------------------
    ln_D  = math.log(T_source + f_p) + f_s
    median = math.exp(ln_D)   # s

    # ------------------------------------------------------------------
    # Standard deviations (equations 14–15)
    # ------------------------------------------------------------------
    # tau — inter-event
    if Mw < 6.5:
        tau = C['tau1']
    elif Mw < 7.0:
        tau = C['tau1'] + (C['tau2'] - C['tau1']) * (Mw - 6.5) / 0.5
    else:
        tau = C['tau2']

    # phi — intra-event
    if Mw < 5.5:
        phi = C['phi1']
    elif Mw < 5.75:
        phi = C['phi1'] + (C['phi2'] - C['phi1']) * (Mw - 5.5) / 0.25
    else:
        phi = C['phi2']

    sigma = math.sqrt(tau**2. + phi**2.)

    return median, sigma, tau, phi
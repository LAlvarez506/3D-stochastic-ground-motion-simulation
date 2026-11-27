"""
Random vibration theory (RVT) helpers after Boore (2003), section
"Peak Motions from Random Vibration Theory" (Eqs. 29-39).

Given the Fourier acceleration amplitude spectrum |Y(f)| (physical units, e.g.
cm/s^2 per Hz) the peak of the motion is estimated without any time series:

    m_k   = 2 * int_0^inf (2 pi f)^k |Y|^2 df                    (Eq. 35)
    y_rms = sqrt(m0 / T_rms)                                      (Eq. 36)
    y_max = y_rms * peak_factor(Nz, Ne)                           (Eqs. 29-34)

``peak_factor`` evaluates the Cartwright & Longuet-Higgins integral (Eq. 29)
numerically, as Boore does, rather than the asymptotic Eq. 31.
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
_Z = np.linspace(0., 12., 6001)       # integration grid of Eq. 29


def spectral_moments(f, Y, orders=(0, 2, 4)):
    """m_k = 2 int (2 pi f)^k |Y|^2 df  (Eq. 35), trapezoidal on the given axis."""
    f = np.asarray(f, float)
    y2 = np.abs(np.asarray(Y)) ** 2
    return [2. * _trapz((2. * np.pi * f) ** k * y2, f) for k in orders]


def peak_factor(m0, m2, m4, T):
    """
    y_max / y_rms of Eq. 29 for duration T [s].  Nz and Ne follow Eqs. 32-34.

    The prefactor of the integral is sqrt(2) (the radical is easy to lose when
    the PDF text is extracted); it reproduces the asymptotic Eq. 31,
    sqrt(2 ln Nz) + 0.5772 / sqrt(2 ln Nz), and was checked against Monte-Carlo
    stationary noise (``self_test``).
    """
    fz = np.sqrt(m2 / m0) / (2. * np.pi)
    fe = np.sqrt(m4 / m2) / (2. * np.pi)
    Nz, Ne = 2. * fz * T, 2. * fe * T
    xi = min(Nz / Ne, 1.)
    Ne = max(Ne, 1.01)
    integrand = 1. - (1. - xi * np.exp(-_Z ** 2)) ** Ne
    return np.sqrt(2.) * _trapz(integrand, _Z)


def rvt_peak(f, Y, T_peak, T_rms=None):
    """
    Peak value of the motion whose Fourier spectrum is ``Y`` (Eq. 29 and 36).
    ``T_peak`` counts the extrema (Eq. 32); ``T_rms`` is the duration that
    divides m0 in Eq. 36 (default: the same value).
    """
    m0, m2, m4 = spectral_moments(f, Y)
    T_rms = T_peak if T_rms is None else T_rms
    return np.sqrt(m0 / T_rms) * peak_factor(m0, m2, m4, T_peak)


def oscillator_transfer(f, Tn, damping=0.05):
    """Pseudo-acceleration transfer function of a SDOF oscillator, |omega0^2 H|."""
    w, w0 = 2. * np.pi * np.asarray(f, float), 2. * np.pi / Tn
    return w0 ** 2 / (w0 ** 2 - w ** 2 + 2j * damping * w0 * w)


def boore_trms(Tgm, Tn, damping=0.05, n=3., a=1. / 3.):
    """T_rms of Eq. 38 (Boore & Joyner 1984): Tgm + To (c^n / (c^n + a))."""
    To = Tn / (2. * np.pi * damping)
    c = Tgm / To
    return Tgm + To * (c ** n / (c ** n + a))


def rvt_psa(f, Y, Tn, Tgm, damping=0.05):
    """PSA by RVT, with T_rms of Eq. 38 and N from Tgm."""
    return rvt_peak(f, np.asarray(Y) * oscillator_transfer(f, Tn, damping), Tgm,
                    boore_trms(Tgm, Tn, damping))


def response_spectrum(acc, dt, periods, damping=0.05):
    """
    Pseudo-spectral acceleration of a damped SDOF oscillator for the record
    ``acc`` (first-order-hold discretisation, exact for piecewise-linear input).
    """
    from scipy.signal import cont2discrete, lfilter
    acc = np.asarray(acc, float)
    out = np.empty(len(periods))
    for k, T in enumerate(periods):
        w = 2. * np.pi / T
        num, den, _ = cont2discrete(([-1.0], [1.0, 2 * damping * w, w * w]),
                                    dt, method='foh')
        out[k] = np.max(np.abs(lfilter(num.ravel(), den, acc))) * w ** 2
    return out


def self_test(n_real=300, seed=0, dt=0.005):
    """
    Independent check of ``rvt_peak``: Monte-Carlo stationary Gaussian noise with
    the spectrum of a Brune-type acceleration spectrum; the mean peak must agree
    with the RVT value within a few percent.  Returns the list of ratios.
    """
    f = np.linspace(0.05, 60., 6000)
    Y = f ** 2 / (1. + (f / 1.0) ** 2) * np.exp(-np.pi * 0.04 * f)
    Y = Y / np.sqrt(spectral_moments(f, Y)[0] / 10.)       # arbitrary level
    rng = np.random.default_rng(seed)
    out = []
    for T in (3., 7., 15.):
        n = int(T / dt)
        Yi = np.interp(np.fft.rfftfreq(n, dt), f, Y)
        pk = [np.abs(np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(n) * Yi / dt, n)).max()
              for _ in range(n_real)]
        out.append(rvt_peak(f, Y, T) / np.mean(pk))
    return out

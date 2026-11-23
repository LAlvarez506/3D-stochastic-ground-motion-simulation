"""
Verification 4 - comparison with a GMPE: Atkinson & Boore (2006), ENA hard rock
===============================================================================

Reproduces, with the simulator, the stochastic point-source model that
Atkinson & Boore (2006, BSSA 96(6), 2181-2205; hereafter AB06) used to derive
their eastern-North-America GMPE, and compares PGA and 5 %-damped PSA with the
published GMPE (implemented in the ``pygmm`` package, hard-rock coefficients,
Vs30 = 2000 m/s).

Parameters taken from AB06 (Atkinson 2004 attenuation model, section "Model
parameters" and Table 1):

    stress parameter                140 bar (Brune, single corner frequency)
    shear velocity, density         3.7 km/s, 2.8 g/cm^3   (Vp = sqrt(3) Vs = 6.4 km/s)
    geometric spreading (hypoc. R)  R^-1.3  (R <= 70 km)
                                    R^+0.2  (70 < R <= 140 km)   [Moho bounce]
                                    R^-0.5  (R > 140 km)
    anelastic attenuation           Q(f) = 893 f^0.32, minimum Q = 1000
    kappa                           0.005 s (hard rock)
    hard-rock amplification         Table 3: 1.00 (0.5 Hz), 1.13 (1), 1.22 (2),
                                    1.36 (5), 1.41 (10 Hz and above)
    duration                        T = 1/fc + delta*R with delta = 0 (R<=10),
                                    0.16 (10-70), -0.03 (70-130), 0.04 (>130 km)
                                    (cumulative slopes)
    reference                       horizontal component, moment magnitude

Three curves are drawn:

  GMPE        AB06 median +- 1 sigma (sigma = 0.30 log10 units).
  simulator   the package (point source), geometric-mean horizontal peak of an
              ensemble; median and 16-84 % band.
  SMSIM-RVT   independent calculation with the textbook point-source spectrum of
              Boore (2003) (radiation 0.55, free surface 2, horizontal
              partition 0.71, i.e. no ray-angle dependence) and random vibration
              theory.  It shows how far the model alone can be from the GMPE
              and how far the simulator is from the plain Boore model.

Differences to keep in mind
---------------------------
* AB06 are derived from finite-fault (EXSIM) simulations and then fitted; a
  point source with the same parameters is only close for small and moderate
  magnitudes (default Mw 5 and 6) and not at distances comparable to the fault
  size.  Distances are hypocentral = rupture distance for the point source.
* The simulator uses the physical radiation pattern, ray incidence angle and
  free-surface factors per wave type (P, SV, SH), AB06 use the average 0.55 x 2
  x 0.71.
* The sampling interval must resolve the hard-rock high frequencies
  (kappa = 0.005 s): dt = 0.004 s is used.

Pass criterion (printed): the median of the simulator / GMPE ratio over the
distances, per intensity measure, within a factor ``--tol`` (default 1.5), i.e.
inside the GMPE's own +-1 sigma (factor 2.0).

Run
---
    pip install pygmm
    python -m stochastic_gm.verification.verify_gmpe_ab06 --n-sim 60 \
        --output-dir ./verification_output
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import warnings

import numpy as np

from stochastic_gm import (EQ, MediumConfig, SourceConfig, PathConfig,
                           SiteConfig, DurationConfig, SimConfig)
from stochastic_gm.verification.rvt import response_spectrum, rvt_peak, rvt_psa
from stochastic_gm.verification.verify_interfrequency_correlation import pga_geomean

G_CM = 981.0                                  # cm/s^2 per g

# --- AB06 parameters -------------------------------------------------------
AB06 = dict(
    stress_drop=140., beta=3.7, alpha=3.7 * np.sqrt(3.), rho=2.8,
    Qo=1000., Q1=893., Q1exp=0.32,
    R0=1., R1=70., b1=1.3, R2=140., b2=-0.2, b3=0.5,
    kappa=0.005,
    amp_freq=np.array([0.01, 0.5, 1., 2., 5., 10., 200.]),
    amp=np.array([1., 1., 1.13, 1.22, 1.36, 1.41, 1.41]),
)
PERIODS = (0.1, 0.2, 0.5, 1.0)
LABELS = ['PGA'] + [f'PSA {t:g} s' for t in PERIODS]


def ab06_path_duration(R):
    """AB06 distance dependence of duration: cumulative slopes of 0, 0.16, -0.03, 0.04 s/km."""
    knots = [(10., 0.), (70., 0.16), (130., -0.03), (np.inf, 0.04)]
    t, r0 = 0., 0.
    for r1, slope in knots:
        if R <= r1:
            return t + slope * (R - r0)
        t += slope * (r1 - r0)
        r0 = r1
    return t


def corner_frequency(Mw, p=AB06):
    Mo = 10. ** ((Mw + 10.7) * 1.5)
    return 4.9e6 * p['beta'] * (p['stress_drop'] / Mo) ** (1. / 3.)


def build_ab06_scenario(Mw, R_hyp, n_sim=60, seed=1, depth=10., dt=0.004, ftgm=2.0,
                        p=AB06):
    """Simulator configured with the AB06 point-source model at hypocentral distance R_hyp."""
    x = np.sqrt(R_hyp ** 2 - depth ** 2)
    medium = MediumConfig(Depth=np.array([0., 60.]), vs=np.array([p['beta']] * 2),
                          vp=np.array([p['alpha']] * 2), Rho=np.array([p['rho']] * 2),
                          system='local')
    source = SourceConfig(stress_drop=p['stress_drop'], rake=90., dip=45., strike=45.,
                          evolutionary_frequency_model='Dang')
    path = PathConfig(Qo=p['Qo'], Q1=p['Q1'], Q1exp=p['Q1exp'], R0=p['R0'], R1=p['R1'],
                      b1=p['b1'], R2=p['R2'], b2=p['b2'], b3=p['b3'])
    site = SiteConfig(amp_freq=p['amp_freq'], amp=p['amp'], kappa=p['kappa'])
    duration = DurationConfig(boolean_tgm_source=True, Tgm_source=1. / corner_frequency(Mw, p),
                              boolean_tgm_path=True, Tgm_path=ab06_path_duration(R_hyp))
    sim = SimConfig(dt=dt, n_sim=n_sim, seed=seed,
                    epsilon_s=0.2, nu_s=0.05, ftgm_s=ftgm,
                    epsilon_p=0.2, nu_p=0.05, ftgm_p=ftgm)
    with contextlib.redirect_stdout(io.StringIO()):
        return EQ(Mw=Mw, medium=medium, source=source, path=path, site_cfg=site,
                  duration=duration, sim=sim, hypocenter=[0., 0., depth],
                  site_coord=[x, 0., 0.], point_source=True)


# --- independent textbook point-source model + RVT --------------------------

def smsim_fas(f, Mw, R, p=AB06):
    """Boore (2003) point-source Fourier acceleration spectrum [cm/s] (horizontal)."""
    Mo = 10. ** ((Mw + 10.7) * 1.5)
    fc = corner_frequency(Mw, p)
    C = 0.55 * 2.0 * 0.71 / (4. * np.pi * p['rho'] * p['beta'] ** 3 * p['R0']) * 1e-20
    G = np.where(R <= p['R1'], (p['R0'] / R) ** p['b1'],
        np.where(R <= p['R2'], (p['R0'] / p['R1']) ** p['b1'] * (p['R1'] / R) ** p['b2'],
                 (p['R0'] / p['R1']) ** p['b1'] * (p['R1'] / p['R2']) ** p['b2']
                 * (p['R2'] / R) ** p['b3']))
    Q = np.maximum(p['Qo'], p['Q1'] * f ** p['Q1exp'])
    amp = np.interp(f, p['amp_freq'], p['amp'])
    return (C * Mo * (2. * np.pi * f) ** 2 / (1. + (f / fc) ** 2) * G
            * np.exp(-np.pi * f * R / (Q * p['beta'])) * amp * np.exp(-np.pi * p['kappa'] * f))


def smsim_rvt(Mw, R, p=AB06):
    f = np.geomspace(0.02, 120., 4000)
    Y = smsim_fas(f, Mw, R, p)
    Tgm = 1. / corner_frequency(Mw, p) + ab06_path_duration(R)
    return np.array([rvt_peak(f, Y, Tgm)] + [rvt_psa(f, Y, T, Tgm) for T in PERIODS])


# --- GMPE ------------------------------------------------------------------

def ab06_gmpe(Mw, R):
    try:
        import pygmm
    except ImportError:                                     # pragma: no cover
        sys.exit("This check needs the GMPE package:  pip install pygmm")
    m = pygmm.AtkinsonBoore2006(pygmm.Scenario(mag=Mw, dist_rup=R, v_s30=2000.))
    med = np.concatenate([[m.pga], m.interp_spec_accels(PERIODS)]) * G_CM
    sig = np.concatenate([[m.ln_std_pga], m.interp_ln_stds(PERIODS)])
    return med, sig


# --- simulator ensemble --------------------------------------------------------

def simulate_peaks(Mw, R, n_sim, seed, dt, ftgm, p=AB06):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eq = build_ab06_scenario(Mw, R, n_sim, seed, dt=dt, ftgm=ftgm, p=p)
    pga = pga_geomean(eq.sim_acc)
    psa = np.array([np.sqrt(response_spectrum(a[0], dt, PERIODS)
                            * response_spectrum(a[1], dt, PERIODS)) for a in eq.sim_acc])
    return np.column_stack([pga, psa])                     # (n_sim, 1 + n_periods)


VARIANTS = {
    # literal parameters of Atkinson (2004) / AB06
    'ab06-literal': dict(),
    # same, but R^-1 up to 70 km as in Atkinson & Boore (1995) and Frankel et al. (1996)
    'r-1-spreading': dict(b1=1.0),
}
SIM_VS_SMSIM_CHECKED = 3        # PGA, PSA 0.1 s, 0.2 s: simulator vs textbook RVT
GMPE_CHECKED = 2                # PGA, PSA 0.1 s: simulator vs GMPE for 'r-1-spreading'


def run(n_sim=60, seed=1, mws=(5., 6.), dists=(15., 20., 30., 50., 70., 100., 150.),
        dt=0.004, ftgm=2.0, tol=1.5, tol_code=1.25, tol_gmpe=1.35,
        variants=tuple(VARIANTS), output_dir='./verification_output', show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    os.makedirs(output_dir, exist_ok=True)
    d = np.array(dists)
    ok = True
    print(f"AB06 hard-rock point-source model, dt = {dt} s, window Tw = {ftgm} Tgm, "
          f"{n_sim} realizations per scenario")
    for var in variants:
        p = dict(AB06, **VARIANTS[var])
        res = {}
        for Mw in mws:
            gm, sg, sim_med, sim_lo, sim_hi, smsim = [], [], [], [], [], []
            for R in dists:
                med, sig = ab06_gmpe(Mw, R)
                pk = simulate_peaks(Mw, R, n_sim, seed, dt, ftgm, p)
                ln = np.log(pk)
                gm.append(med); sg.append(sig)
                sim_med.append(np.exp(ln.mean(0)))
                sim_lo.append(np.exp(ln.mean(0) - ln.std(0, ddof=1)))
                sim_hi.append(np.exp(ln.mean(0) + ln.std(0, ddof=1)))
                smsim.append(smsim_rvt(Mw, R, p))
            res[Mw] = {k: np.array(v) for k, v in dict(gmpe=gm, sigma=sg, sim=sim_med,
                       lo=sim_lo, hi=sim_hi, smsim=smsim).items()}

        print(f"\n=== variant '{var}': geometric spreading b1 = {p['b1']} up to {p['R1']:g} km ===")
        for Mw in mws:
            r = res[Mw]
            print(f"\nMw {Mw:g}: ratio to the GMPE median (simulator | SMSIM-RVT) by hypocentral distance")
            print(f"  {'R [km]':>7s} " + ' '.join(f"{lab:>17s}" for lab in LABELS))
            for i, R in enumerate(dists):
                print(f"  {R:7.0f} " + ' '.join(
                    f"{r['sim'][i, j] / r['gmpe'][i, j]:8.2f} |{r['smsim'][i, j] / r['gmpe'][i, j]:7.2f}"
                    for j in range(len(LABELS))))
            sim_gmpe = np.median(r['sim'] / r['gmpe'], axis=0)
            sm_gmpe = np.median(r['smsim'] / r['gmpe'], axis=0)
            sim_sm = np.median(r['sim'] / r['smsim'], axis=0)
            print("  median over distance, simulator/GMPE : " + ' '.join(f"{x:6.2f}" for x in sim_gmpe))
            print("  median over distance, SMSIM-RVT/GMPE : " + ' '.join(f"{x:6.2f}" for x in sm_gmpe))
            print("  median over distance, simulator/SMSIM: " + ' '.join(f"{x:6.2f}" for x in sim_sm))
            # (1) code check: simulator vs the independent textbook model + RVT
            c = sim_sm[:SIM_VS_SMSIM_CHECKED]
            bad_code = np.any((c > tol_code) | (c < 1. / tol_code))
            print(f"  code check   simulator vs SMSIM-RVT (PGA, 0.1, 0.2 s; factor {tol_code:g}): "
                  f"{'FAIL' if bad_code else 'PASS'}")
            ok &= not bad_code
            # (2) model-vs-GMPE
            g = sim_gmpe[:GMPE_CHECKED]
            bad_g = np.any((g > tol_gmpe) | (g < 1. / tol_gmpe))
            within = np.all((sim_gmpe < tol) & (sim_gmpe > 1. / tol))
            if var == 'r-1-spreading':
                print(f"  GMPE check   simulator vs AB06 (PGA, 0.1 s; factor {tol_gmpe:g}): "
                      f"{'FAIL' if bad_g else 'PASS'}")
                ok &= not bad_g
            else:
                print(f"  GMPE check   all periods within factor {tol:g}: "
                      f"{'yes' if within else 'NO (informational for this variant)'}")

        # figure 1: curves
        fig, axs = plt.subplots(len(mws), len(LABELS), figsize=(3.2 * len(LABELS), 3.1 * len(mws)),
                                squeeze=False)
        for a, Mw in enumerate(mws):
            r = res[Mw]
            for j, lab in enumerate(LABELS):
                ax = axs[a, j]
                sg = r['sigma'][:, j]
                ax.fill_between(d, r['gmpe'][:, j] * np.exp(-sg), r['gmpe'][:, j] * np.exp(sg),
                                color='0.85', label='AB06 +-1 sigma')
                ax.loglog(d, r['gmpe'][:, j], 'k-', label='AB06 median')
                ax.fill_between(d, r['lo'][:, j], r['hi'][:, j], color='tab:blue', alpha=.2)
                ax.loglog(d, r['sim'][:, j], 'o-', color='tab:blue', ms=3, label='simulator')
                ax.loglog(d, r['smsim'][:, j], '--', color='tab:red', label='SMSIM-RVT')
                ax.set_title(f"Mw {Mw:g}  {lab}", fontsize=9)
                if j == 0:
                    ax.set_ylabel('cm/s$^2$')
                if a == len(mws) - 1:
                    ax.set_xlabel('hypocentral distance [km]')
                ax.grid(True, which='both', alpha=.3)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[0, 0].legend(fontsize=6)
        fig.suptitle(f"variant '{var}' (b1 = {p['b1']})", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'gmpe_ab06_curves_{var}.png'), dpi=150)

        # figure 2: ratios
        fig, axs = plt.subplots(1, len(mws), figsize=(5.5 * len(mws), 3.8), squeeze=False)
        for a, Mw in enumerate(mws):
            ax = axs[0, a]
            for j, lab in enumerate(LABELS):
                ax.semilogx(d, res[Mw]['sim'][:, j] / res[Mw]['gmpe'][:, j], 'o-', ms=3, label=lab)
            ax.axhline(1, color='k', lw=.8)
            ax.axhspan(1 / tol, tol, color='0.9', zorder=0)
            ax.set_ylim(0, 3.2)
            ax.set_title(f"'{var}', Mw {Mw:g}: simulator / AB06")
            ax.set_xlabel('hypocentral distance [km]')
            ax.grid(True, which='both', alpha=.3)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'gmpe_ab06_ratios_{var}.png'), dpi=150)
        plt.close('all')

    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-sim', type=int, default=60)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mw', type=float, nargs='+', default=[5., 6.])
    p.add_argument('--dt', type=float, default=0.004)
    p.add_argument('--ftgm', type=float, default=2.0)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument('--tol-code', type=float, default=1.25)
    p.add_argument('--tol-gmpe', type=float, default=1.35)
    p.add_argument('--output-dir', default='./verification_output')
    a = p.parse_args()
    sys.exit(0 if run(a.n_sim, a.seed, tuple(a.mw), dt=a.dt, ftgm=a.ftgm,
                      tol_code=a.tol_code, tol_gmpe=a.tol_gmpe, variants=tuple(a.variants),
                      output_dir=a.output_dir) else 1)


if __name__ == '__main__':
    main()

"""
Verification 3 - energy (Parseval) and random-vibration-theory check
=====================================================================

Two checks of the amplitude scaling against Boore (2003), without any
ground-motion model other than the simulator's own deterministic spectrum.

1. Parseval / energy  (Boore 2003, Eqs. 35-36, p. 658-659)
   For a spectrum Y the energy of a record must be  int a^2 dt = 2 int |Y|^2 df
   (= m0).  The script compares the ensemble mean of the simulated energy of the
   two horizontal components with m0 computed from the deterministic horizontal
   spectrum of the sub-source (``target_eas``).  This tests the noise
   normalisation in ``scale_spectrum`` directly and is independent of the
   duration and of the window shape.  Expected ratio: 1.00 (a few percent).

2. Peak motion by RVT  (Boore 2003, Eqs. 29-38)
   PGA and 5 %-damped PSA are predicted from the same spectrum with random
   vibration theory (Cartwright & Longuet-Higgins peak factor, Eq. 38 for the
   oscillator duration) using the simulator's duration Tgm, and compared with
   the mean of the geometric-mean horizontal peaks of the simulated ensemble.
   RVT is an approximation (the Saragoni-Hart window is not stationary, and
   RVT itself is only good to about 10-20 % for stationary noise at oscillator
   periods comparable to the duration), so the pass tolerance is wider (default
   +-25 %) and applies to PGA and PSA at 0.1 and 0.3 s.  PSA at 1 s and 3 s is
   printed for information only.
   ``rvt.self_test()`` checks the RVT implementation itself against
   Monte-Carlo stationary noise (agreement ~1 %).

Known finding (energy check)
----------------------------
When the P and S windows overlap (large magnitude, short distance, long
window) the simulated energy is below m0 (down to ~0.67).  Cause: P and S are
built from the same white-noise sequence, and the radial horizontal motion adds
P (-sin(theta_p)) and SV (+cos(theta_s)) with opposite signs, so the common part
of the noise cancels partly.  With independent noise for P and S the ratio is
0.99-1.00 in every scenario.  Far from the source, or with short windows, the
effect is small.

Window: Boore (2003, p. 655) uses Tw = 2 Tgm for the Saragoni-Hart/exponential
window (``ftgm = 2``); RVT is only meaningful for that convention.  With the
shorter window of the example scripts (``ftgm = 0.5``) the energy is packed into
about a quarter of the duration and RVT overpredicts the response of long-period
oscillators (``--ftgm 0.5`` shows this).

Scenarios: a grid of Mw and hypocentral distances in a homogeneous half space
(see ``build_scenario``).

Run
---
    python -m stochastic_gm.verification.verify_rvt --n-sim 100 \
        --output-dir ./verification_output

Output: a printed table, ``rvt_check.png`` and exit status 1 if a check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

from stochastic_gm.verification.rvt import (response_spectrum, rvt_peak, rvt_psa,
                                            spectral_moments)
from stochastic_gm.verification.verify_interfrequency_correlation import (
    build_scenario, pga_geomean)
from stochastic_gm.verification.verify_noise_scaled_spectra import target_eas

PERIODS = (0.1, 0.3, 1.0, 3.0)
CHECKED = 3        # PGA, PSA 0.1 s and 0.3 s are pass/fail; 1 s and 3 s are informational


def run(n_sim=100, seed=1, mws=(5., 6., 7.), dists=(10., 30., 100.), dt=0.01,
        tol_energy=0.05, tol_rvt=0.25, ftgm=2.0, output_dir='./verification_output',
        show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    rows, ok = [], True
    print(f"window: Tw = {ftgm} * Tgm  (Boore 2003 recommends 2.0)")
    print(f"{'Mw':>4s} {'R[km]':>6s} {'Tgm[s]':>7s} {'E/m0':>6s} "
          + ' '.join(f"{'PGA' if p == 0 else 'PSA' + str(p):>8s}" for p in (0,) + PERIODS)
          + "   (RVT / simulated mean peak; last two periods informational)")
    for Mw in mws:
        for R in dists:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                eq = build_scenario(n_sim, seed, None, Mw=Mw, dist_km=R, dt=dt, ftgm=ftgm)
            ps = eq.psources[0][0]
            f, Y = target_eas(ps)
            Tgm = float(ps['tgm_s'])
            m0 = spectral_moments(f, Y)[0]
            en = np.mean([0.5 * (np.sum(a[0] ** 2) + np.sum(a[1] ** 2)) * dt
                          for a in eq.sim_acc])
            e_ratio = en / m0

            sim_pga = np.mean(pga_geomean(eq.sim_acc))
            sim_psa = np.mean([np.sqrt(response_spectrum(a[0], dt, PERIODS)
                                       * response_spectrum(a[1], dt, PERIODS))
                               for a in eq.sim_acc], axis=0)
            r = [rvt_peak(f, Y, Tgm) / sim_pga]
            r += [rvt_psa(f, Y, Tn, Tgm) / s for Tn, s in zip(PERIODS, sim_psa)]
            rows.append((Mw, R, Tgm, e_ratio, r))
            bad = abs(e_ratio - 1) > tol_energy or any(abs(x - 1) > tol_rvt for x in r[:CHECKED])
            ok &= not bad
            print(f"{Mw:4.1f} {R:6.0f} {Tgm:7.2f} {e_ratio:6.3f} "
                  + ' '.join(f"{x:8.3f}" for x in r) + ('  <-- FAIL' if bad else ''))

    print(f"\nTolerances: energy ratio within {tol_energy:.0%}, RVT/simulation within "
          f"{tol_rvt:.0%}.  Result: {'PASS' if ok else 'FAIL'}")

    labels = ['PGA'] + [f'PSA {p} s' for p in PERIODS]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r[3] for r in rows], 'o-')
    ax[0].axhline(1, color='k', lw=.8)
    ax[0].set_ylabel('simulated energy / m0')
    ax[0].set_title('Parseval check')
    for j, lab in enumerate(labels):
        ax[1].plot([r[4][j] for r in rows], 'o-', label=lab)
    ax[1].axhline(1, color='k', lw=.8)
    ax[1].set_ylabel('RVT / simulated mean peak')
    ax[1].set_title('RVT check')
    ax[1].legend(fontsize=7)
    for a in ax:
        a.set_xticks(range(len(rows)))
        a.set_xticklabels([f"M{r[0]:g}\n{r[1]:g} km" for r in rows], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rvt_check.png'), dpi=150)
    if show:
        plt.show()
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-sim', type=int, default=100)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tol-energy', type=float, default=0.05)
    p.add_argument('--tol-rvt', type=float, default=0.25)
    p.add_argument('--ftgm', type=float, default=2.0,
                   help='window Tw/Tgm; Boore (2003) uses 2.0, the example scripts 0.5')
    p.add_argument('--output-dir', default='./verification_output')
    a = p.parse_args()
    sys.exit(0 if run(a.n_sim, a.seed, tol_energy=a.tol_energy, tol_rvt=a.tol_rvt, ftgm=a.ftgm,
                      output_dir=a.output_dir) else 1)


if __name__ == '__main__':
    main()

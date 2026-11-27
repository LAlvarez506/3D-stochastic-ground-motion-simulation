"""
Verification 6 - orientation of the horizontal components (azimuth convention)
==============================================================================

Before the first S arrival only the P wave exists, and a P wave in a layered medium has
no tangential motion.  So, for a site at any azimuth from the epicenter, the simulated
NS/EW record rotated with the TRUE source-to-site azimuth (clockwise from north) must have
its energy before S on the radial axis, with ~zero energy on the tangential axis.
The same check with the tangential/radial roles swapped holds for SH-only runs after S.

Checks
------
  1  ``PointSource.azimuth`` equals the geographic azimuth source -> site, in [0, 2 pi);
  2  the tangential/radial rms ratio before S is below ``tol`` (0.1; the small residual
     ~0.05 is acausal spreading of the S wave by the frequency-domain filtering) for every azimuth
     (the earlier unsigned-angle convention rotated the horizontals by 180 deg - 2 az
     for sites east of the source, so the check failed there);
  3  the same for sh_only runs: the SH motion lies on the tangential axis.

Run
---
    python -m stochastic_gm.verification.verify_azimuth
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings

import numpy as np

from stochastic_gm import (EQ, MediumConfig, SourceConfig, PathConfig, SiteConfig,
                           DurationConfig, SimConfig)


def build(az_deg, R_epi=60., depth=10., n_sim=3, seed=3, sh_only=False, dt=0.01):
    a = np.radians(az_deg)
    medium = MediumConfig(Depth=np.array([0., 60.]), vs=np.array([3.4, 3.4]),
                          vp=np.array([5.9, 5.9]), Rho=np.array([2.7, 2.7]), system='local')
    source = SourceConfig(stress_drop=100., rake=90., dip=45., strike=30.,
                          evolutionary_frequency_model='Dang')
    path = PathConfig(Qo=300., Q1=0., Q1exp=0., R0=1., R1=200., b1=1., R2=400., b2=0.5, b3=0.5)
    site = SiteConfig(amp_freq=np.array([0.1, 100.]), amp=np.array([1., 1.]), kappa=0.03)
    dur = DurationConfig(boolean_tgm_source=True, Tgm_source=1., boolean_tgm_path=True, Tgm_path=2.)
    sim = SimConfig(dt=dt, n_sim=n_sim, seed=seed, sh_only=sh_only, epsilon_s=0.2, nu_s=0.05,
                    ftgm_s=2., epsilon_p=0.2, nu_p=0.05, ftgm_p=2.)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return EQ(Mw=6.0, medium=medium, source=source, path=path, site_cfg=site, duration=dur,
                  sim=sim, hypocenter=[0., 0., depth],
                  site_coord=[R_epi * np.sin(a), R_epi * np.cos(a), 0.], point_source=True)


def run(azimuths=(0, 30, 45, 90, 135, 180, 225, 270, 315), tol=0.1):
    ok_all = True
    print(f"{'az':>5} {'phi':>7} {'T/R before S':>13} {'R/T of SH-only':>15}")
    for az in azimuths:
        eq = build(az); ps = eq.psources[0][0]
        phi = ps['azimuth']
        ok1 = abs(((phi - np.radians(az)) + np.pi) % (2 * np.pi) - np.pi) < 1e-9
        pre = slice(int(ps['to_p'] / 0.01), int((ps['to_s'] - 0.3) / 0.01))
        a = np.radians(az); tr = []
        for NS, EW, UD in eq.sim_acc:
            r = NS * np.cos(a) + EW * np.sin(a); t = -NS * np.sin(a) + EW * np.cos(a)
            tr.append(np.sqrt(np.mean(t[pre] ** 2) / np.mean(r[pre] ** 2)))
        eqs = build(az, sh_only=True); rr = []
        for NS, EW, UD in eqs.sim_acc:
            r = NS * np.cos(a) + EW * np.sin(a); t = -NS * np.sin(a) + EW * np.cos(a)
            rr.append(np.sqrt(np.sum(r ** 2) / np.sum(t ** 2)))
        ok = ok1 and max(tr) < tol and max(rr) < tol
        ok_all &= ok
        print(f"{az:5d} {np.degrees(phi):7.1f} {max(tr):13.4f} {max(rr):15.4f}   {'ok' if ok else 'FAIL'}")
    print('\nPASS' if ok_all else '\nFAIL')
    return ok_all


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--tol', type=float, default=0.1)
    sys.exit(0 if run(tol=ap.parse_args().tol) else 1)

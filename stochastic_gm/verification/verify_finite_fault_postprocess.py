"""
Verification 5 - correlated noise on a finite fault by post-processing
======================================================================

A synthetic planar fault (default Mw 6.5, 20 x 10 km, 8 x 4 sub-sources, rupture
speed 2.8 km/s, site 25 km from the fault) is simulated as an ensemble of
summed records.  Three versions are compared:

  baseline        original independent noise per sub-source (no correlation);
  per-subsource   ``SimConfig.corr_noise`` on a finite fault: every sub-source
                  gets its own correlated epsilon, then the records are summed
                  (the behaviour that triggers the warning in ``EQ``);
  post-processed  baseline records + ``postprocess.postprocess_eq``: ONE
                  correlated epsilon imposed on the summed record.

Checks (target: the BA19 matrix / sigma(f; Mw) of the generator):

  1  correlation structure of the smoothed EAS (b = 188.5) of the ensemble,
     as in verification 1 (Fig. 5 of BA19): RMSE against the model, passing if
     <= 2 / sqrt(n_sim) (about twice the sampling noise);
  2  standard deviation of ln EAS against sigma(f; Mw), mean ratio within 10 %;
  3  energy: mean energy of the post-processed horizontals / original (1.0 within
     3 standard errors for normalization 'expected'; the ratio is heavy-tailed
     because the BA19 residual is strongly correlated in frequency) and rms EAS ratio;
  4  components: the NS/EW spectral ratio is unchanged, the vertical is
     identical when apply_to = 'horizontal'.

Run
---
    python -m stochastic_gm.verification.verify_finite_fault_postprocess --n-sim 800 \
        --output-dir ./verification_output
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

from stochastic_gm import (EQ, MediumConfig, SourceConfig, PathConfig, SiteConfig,
                           DurationConfig, SimConfig, CorrelatedNoiseConfig)
from stochastic_gm.postprocess import postprocess_eq
from stochastic_gm.verification.verify_interfrequency_correlation import (
    COND_COLORS, COND_FREQS, binned_comparison, fisher_ci, sample_statistics,
    smoothed_eas)


def planar_fault(Mw_unused=None, L=20., W=10., dip=60., depth_top=3., nx=8, ny=4,
                 vr=2.8, hypo=(0.25, 0.5)):
    """Sub-source dicts (local km; strike along +y, dipping towards +x) and hypocenter."""
    dx, dy = L / nx, W / ny
    d = np.radians(dip)
    subs = []
    for i in range(nx):
        for j in range(ny):
            s, w = (i + .5) * dx - L / 2, (j + .5) * dy
            subs.append(dict(centroid=[w * np.cos(d), s, depth_top + w * np.sin(d)],
                             slip=1., rupt_t=0., dip=dip, rake=90.))
    hyp = np.array([hypo[1] * W * np.cos(d), hypo[0] * L - L / 2,
                    depth_top + hypo[1] * W * np.sin(d)])
    for s in subs:
        s['rupt_t'] = float(np.linalg.norm(np.array(s['centroid']) - hyp) / vr)
    return subs, list(hyp)


def build_finite_fault(n_sim, seed, corr_noise=None, Mw=6.5, site_x=25., dt=0.01):
    subs, hyp = planar_fault()
    medium = MediumConfig(Depth=np.array([0., 40.]), vs=np.array([3.46] * 2),
                          vp=np.array([6.0] * 2), Rho=np.array([2.7] * 2), system='local')
    source = SourceConfig(stress_drop=50., rake=90., dip=60., strike=0.,
                          evolutionary_frequency_model='Dang')
    path = PathConfig(Qo=300., Q1=0., Q1exp=0., R0=1., R1=100., b1=1., R2=200., b2=0.5, b3=0.)
    site = SiteConfig(amp_freq=np.array([0.01, 200.]), amp=np.array([1., 1.]), kappa=0.04)
    sim = SimConfig(dt=dt, n_sim=n_sim, seed=seed, epsilon_s=0.2, nu_s=0.05, ftgm_s=2.,
                    epsilon_p=0.2, nu_p=0.05, ftgm_p=2., n_processors=1, corr_noise=corr_noise)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        eq = EQ(Mw=Mw, medium=medium, source=source, path=path, site_cfg=site,
                duration=DurationConfig(path_duration=0.05), sim=sim, hypocenter=hyp,
                site_coord=[site_x, 0., 0.], point_source=False, subsource_list=subs)
        eq.run_simulation()
    return eq


def _rmse(R, Rm):
    iu = np.triu_indices(len(Rm), k=1)
    return float(np.sqrt(np.mean((R - Rm)[iu] ** 2)))


def run(n_sim=800, seed=1, Mw=6.5, correlation_type='within_site', normalization='expected',
        f_min=0.2, f_max=24., stride=2, smoothing_b=10., mode='replace', dt=0.01,
        tol_corr=2.0, tol_sigma=0.10, tol_energy=3.0,
        output_dir='./verification_output', show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    cfg = CorrelatedNoiseConfig(correlation_type=correlation_type, normalization=normalization)
    gen = cfg.build(Mw=Mw)
    inside = np.flatnonzero((gen.nodes >= f_min) & (gen.nodes <= f_max))[::stride]
    centers = gen.nodes[inside]
    R_model = gen.R[np.ix_(inside, inside)]
    sd_target = gen.sigma_nodes[inside]

    t0 = time.time()
    print(f"Finite-fault post-processing verification: n_sim={n_sim}, Mw={Mw}, "
          f"{correlation_type}, normalization={normalization}, mode={mode}, b_ref={smoothing_b}")
    eq0 = build_finite_fault(n_sim, seed, None, Mw=Mw, dt=dt)
    print(f"  baseline ensemble: {len(eq0.sim_acc)} records of {len(eq0.sim_acc[0][0])} samples "
          f"({time.time() - t0:.0f} s)")
    eq_ps = build_finite_fault(n_sim, seed, cfg, Mw=Mw, dt=dt)
    new = postprocess_eq(eq0, cfg, seed=seed, mode=mode, smoothing_b=smoothing_b)

    ens = {'baseline': eq0.sim_acc, 'per-subsource': eq_ps.sim_acc, 'post-processed': new}
    stats = {}
    for k, acc in ens.items():
        eas = smoothed_eas(acc, dt, centers)
        R, sd, mu = sample_statistics(eas)
        stats[k] = dict(eas=eas, R=R, sd=sd, mu=mu)

    # ---- 1. correlation ----------------------------------------------------
    print("\n1. Correlation of the smoothed EAS against the model (RMSE, upper triangle)")
    for k in ens:
        print(f"   {k:<15s} {_rmse(stats[k]['R'], R_model):.3f}")
    print(f"   (sampling noise with n_sim={n_sim}: ~{1 / np.sqrt(n_sim):.3f})")
    print("   binned by fr = |ln(f1/f2)|: model / baseline / per-subsource / post-processed")
    rows = {k: binned_comparison(centers, stats[k]['R'], R_model) for k in ens}
    for j, (lo, hi, cnt, mod, _, _) in enumerate(rows['baseline']):
        print(f"   {lo:4.1f}-{hi:<4.1f} {mod:7.3f} " + ' '.join(f"{rows[k][j][4]:7.3f}" for k in ens))
    rmse_pp = _rmse(stats['post-processed']['R'], R_model)
    ok1 = rmse_pp <= tol_corr / np.sqrt(n_sim)

    # ---- 2. sigma ------------------------------------------------------------
    print("\n2. std of ln EAS: target / baseline / per-subsource / post-processed")
    for f in (0.5, 1., 2., 5., 10., 20.):
        j = np.argmin(abs(centers - f))
        print(f"   {centers[j]:6.2f} Hz {sd_target[j]:6.2f} " +
              ' '.join(f"{stats[k]['sd'][j]:6.2f}" for k in ens))
    ratio_sd = np.mean(stats['post-processed']['sd'] / sd_target)
    print(f"   mean(post-processed / target) over {centers[0]:.2g}-{centers[-1]:.2g} Hz: {ratio_sd:.2f}"
          f"   per-subsource: {np.mean(stats['per-subsource']['sd'] / sd_target):.2f}")
    ok2 = abs(ratio_sd - 1) <= tol_sigma

    # ---- 3. energy -----------------------------------------------------------
    def energy(acc):
        return np.array([0.5 * (np.sum(a[0] ** 2) + np.sum(a[1] ** 2)) * dt for a in acc])
    e_ratio = energy(new) / energy(eq0.sim_acc)
    rms_ratio = np.sqrt(np.mean(stats['post-processed']['eas'] ** 2) /
                        np.mean(stats['baseline']['eas'] ** 2))
    print(f"\n3. energy post/original: mean {e_ratio.mean():.3f}, median {np.median(e_ratio):.3f}, "
          f"std {e_ratio.std():.3f};  rms EAS ratio {rms_ratio:.3f};  median EAS ratio "
          f"{np.exp(np.mean(stats['post-processed']['mu'] - stats['baseline']['mu'])):.3f}")
    if normalization == 'expected':
        print("   (expected energy ratio 1.0; the median EAS ratio is not exp(-sigma^2) because the "
              "baseline's own\n    unsmoothed scatter already lowers its median relative to its rms)")
        se = e_ratio.std(ddof=1) / np.sqrt(len(e_ratio))
        print(f"   mean energy ratio - 1 = {e_ratio.mean() - 1:+.3f}, standard error {se:.3f}")
        ok3 = abs(e_ratio.mean() - 1) <= tol_energy * se
    else:
        ok3 = True

    # ---- 4. components ---------------------------------------------------------
    ud_same = all(np.array_equal(a[2], b[2]) for a, b in zip(new, eq0.sim_acc))
    ns_ew = []
    for a, b in zip(new[:50], eq0.sim_acc[:50]):
        ra = np.sum(a[0] ** 2) / np.sum(a[1] ** 2)
        rb = np.sum(b[0] ** 2) / np.sum(b[1] ** 2)
        ns_ew.append(ra / rb)
    print(f"\n4. NS/EW energy ratio changed by factor {np.median(ns_ew):.3f} (median, 50 records); "
          f"vertical identical: {ud_same}")
    ok4 = ud_same if cfg.apply_to == 'horizontal' else True

    ok = ok1 and ok2 and ok3 and ok4
    print(f"\nChecks: correlation RMSE <= {tol_corr:g}/sqrt(n_sim) = {tol_corr / np.sqrt(n_sim):.3f}: "
          f"{'PASS' if ok1 else 'FAIL'};  "
          f"sigma within {tol_sigma:.0%}: {'PASS' if ok2 else 'FAIL'};  "
          f"mean energy ratio within {tol_energy:g} standard errors of 1: {'PASS' if ok3 else 'FAIL'};  "
          f"vertical untouched: {'PASS' if ok4 else 'FAIL'}   ->  {'PASS' if ok else 'FAIL'}")

    # ---- figures ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    for ax, k in zip(axs, ens):
        for fc, c in zip(COND_FREQS, COND_COLORS):
            j = np.argmin(abs(centers - fc))
            lo, hi = fisher_ci(stats[k]['R'][j], n_sim)
            ax.fill_between(centers, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(centers, stats[k]['R'][j], '-', color=c, lw=1.4)
            ax.plot(centers, R_model[j], '--', color=c, lw=1.1)
        ax.set_xscale('log'); ax.set_xlim(centers[0], centers[-1]); ax.set_ylim(-0.1, 1.02)
        ax.set_xlabel('F1 (Hz)'); ax.grid(alpha=0.3)
        ax.set_title(f"{k}  (RMSE {_rmse(stats[k]['R'], R_model):.3f})", fontsize=10)
    axs[0].set_ylabel(r'$\rho_\epsilon$')
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'{fc:g} Hz')
               for fc, c in zip(COND_FREQS, COND_COLORS)]
    handles += [plt.Line2D([], [], color='gray', ls='-', label='simulated (95 % band)'),
                plt.Line2D([], [], color='gray', ls='--', label='BA19 matrix')]
    fig.legend(handles=handles, fontsize=8, ncol=7, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle(f'Finite fault Mw {Mw}: correlation cross sections ({correlation_type})', fontsize=11)
    f0 = os.path.join(output_dir, 'finite_fault_correlation.png')
    fig.savefig(f0, dpi=150, bbox_inches='tight')

    fig2, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(centers, sd_target, 'k--', lw=1.8, label='target sigma')
    for k in ens:
        ax[0].plot(centers, stats[k]['sd'], lw=1.2, label=k)
    ax[0].set_xscale('log'); ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('std of ln EAS')
    ax[0].legend(fontsize=8)
    ax[1].hist(e_ratio, bins=30, color='tab:blue', alpha=.8)
    ax[1].axvline(1, color='k', lw=.8); ax[1].set_xlabel('energy post / original')
    fig2.tight_layout()
    f1 = os.path.join(output_dir, 'finite_fault_sigma_energy.png')
    fig2.savefig(f1, dpi=150, bbox_inches='tight')
    print(f"\nFigures: {f0}\n         {f1}   (total {time.time() - t0:.0f} s)")
    if show:
        plt.show()
    plt.close('all')
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-sim', type=int, default=800)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mw', type=float, default=6.5)
    p.add_argument('--correlation-type', default='within_site',
                   choices=['within_site', 'total', 'between_site'])
    p.add_argument('--normalization', default='expected',
                   choices=['expected', 'per_realization', 'none'])
    p.add_argument('--mode', default='replace', choices=['replace', 'scale'])
    p.add_argument('--smoothing-b', type=float, default=10.)
    p.add_argument('--output-dir', default='./verification_output')
    a = p.parse_args()
    sys.exit(0 if run(a.n_sim, a.seed, a.mw, a.correlation_type, a.normalization,
                      smoothing_b=a.smoothing_b, mode=a.mode, output_dir=a.output_dir) else 1)


if __name__ == '__main__':
    main()

"""
Verification 1 - recovering the interfrequency correlation structure
====================================================================

Simulates many realisations of one point-source scenario, computes the
effective amplitude spectrum (EAS, Goulet et al. 2018) of each, smooths it
with a Konno-Ohmachi window (b = 188.5, as in Bayless & Abrahamson 2019),
converts ln(EAS) to normalised residuals

    eps_i(f) = ( ln EAS_i(f) - mean_i ln EAS_i(f) ) / std_i ln EAS_i(f)

and compares the sample correlation of eps between all pairs of frequencies
with the target matrix of the paper (default: within-site).  The same
scenario and seed are run without the correlated noise (baseline) and with
it, for each requested normalisation.

Figures (written to --output-dir)
---------------------------------
fig5_style_cross_sections.png   cross sections at 0.2, 0.5, 2, 5, 15 Hz, like
                                Fig. 5 of the paper: model (dashed) against
                                the simulated correlation with its 95 %
                                confidence band (solid); right: baseline.
correlation_contours.png        contours like Fig. 4: model, baseline,
                                simulated, and the difference.
verification_summary.png        sigma(f) and level against the baseline.

Run
---
    python -m stochastic_gm.verification.verify_interfrequency_correlation \
        --n-sim 2000 --output-dir ./verification_output

    # other targets
    python -m stochastic_gm.verification.verify_interfrequency_correlation \
        --correlation-type total

Notes
-----
* Records are about 20 s long, i.e. a frequency resolution of ~0.05 Hz, so the
  default lower limit of the comparison is 0.2 Hz.  Below ~0.2 Hz the sample
  correlation reflects the record length as much as the noise.
* The correlation of ln EAS across realisations of ONE fixed scenario is the
  correlation of the noise alone, which is why the within-site matrix (no
  event or site terms) is the default target.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import warnings

import numpy as np

from stochastic_gm import (EQ, MediumConfig, SourceConfig, PathConfig,
                           SiteConfig, DurationConfig, SimConfig,
                           CorrelatedNoiseConfig)

KO_B = 188.5          # Konno-Ohmachi bandwidth parameter used by PEER / BA19
COND_FREQS = (0.2, 0.5, 2.0, 5.0, 15.0)               # as in Figs. 5, 8, 10 of the papers
COND_COLORS = ('#1f3fd0', '#d01f1f', '#1f8a1f', '#c01fc0', 'k')


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

def build_scenario(n_sim, seed, corr_noise=None, Mw=6.5, dist_km=30.,
                   depth_km=15., dt=0.01, kappa=0.04, ftgm=0.5, independent_ps_noise=True):
    """Homogeneous half-space point-source scenario (edit to taste)."""
    medium = MediumConfig(Depth=np.array([0., 30.]), vs=np.array([3.46, 3.46]),
                          vp=np.array([6.0, 6.0]), Rho=np.array([2.7, 2.7]),
                          system='local')
    source = SourceConfig(stress_drop=50., rake=90., dip=45., strike=45.,
                          evolutionary_frequency_model='Dang')
    path = PathConfig(Qo=300., Q1=0., Q1exp=0., R0=1., R1=100., b1=1.,
                      R2=200., b2=0.5, b3=0.)
    site = SiteConfig(amp_freq=np.array([0.01, 200.]), amp=np.array([1., 1.]),
                      kappa=kappa)
    duration = DurationConfig(path_duration=0.05)
    sim = SimConfig(dt=dt, n_sim=n_sim, seed=seed,
                    epsilon_s=0.2, nu_s=0.05, ftgm_s=ftgm,
                    epsilon_p=0.2, nu_p=0.05, ftgm_p=ftgm,
                    corr_noise=corr_noise,
                    independent_ps_noise=independent_ps_noise)
    with contextlib.redirect_stdout(io.StringIO()):
        eq = EQ(Mw=Mw, medium=medium, source=source, path=path, site_cfg=site,
                duration=duration, sim=sim, hypocenter=[0., 0., depth_km],
                site_coord=[dist_km, 0., 0.], point_source=True)
    return eq


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def konno_ohmachi_matrix(freqs, centers, b=KO_B):
    """Row-normalised log10 Konno-Ohmachi weights, shape (len(centers), len(freqs))."""
    f = freqs[None, :]
    fc = centers[:, None]
    x = b * np.log10(f / fc)
    with np.errstate(invalid='ignore', divide='ignore'):
        W = (np.sin(x) / x) ** 4
    W[~np.isfinite(W)] = 1.0           # x == 0
    return W / W.sum(axis=1, keepdims=True)


def smoothed_eas(sim_acc, dt, centers, pad=8):
    """
    EAS = sqrt( (|FAS_NS|^2 + |FAS_EW|^2) / 2 ), Konno-Ohmachi smoothed, at
    ``centers`` [Hz].  Returns (n_real, n_centers).  Zero padding (x``pad``)
    only densifies the frequency axis.
    """
    n = len(sim_acc[0][0])
    n_fft = int(2 ** np.ceil(np.log2(pad * n)))
    freqs = np.fft.rfftfreq(n_fft, dt)[1:]
    W = konno_ohmachi_matrix(freqs, centers)
    out = np.empty((len(sim_acc), len(centers)))
    for i, (NS, EW, _UD) in enumerate(sim_acc):
        a = np.abs(np.fft.rfft(NS, n_fft))[1:] * dt
        b = np.abs(np.fft.rfft(EW, n_fft))[1:] * dt
        out[i] = W @ np.sqrt(0.5 * (a ** 2 + b ** 2))
    return out


def sample_statistics(eas):
    ln = np.log(eas)
    sd = ln.std(axis=0, ddof=1)
    eps = (ln - ln.mean(axis=0)) / sd
    return np.corrcoef(eps.T), sd, ln.mean(axis=0)


def pga_geomean(sim_acc):
    return np.array([np.sqrt(np.max(np.abs(ns)) * np.max(np.abs(ew)))
                     for ns, ew, _ in sim_acc])


def fisher_ci(r, n, z=1.96):
    """95 % confidence band of a sample correlation (Fisher transform)."""
    zr = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    return np.tanh(zr - z * se), np.tanh(zr + z * se)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def binned_comparison(centers, R_sim, R_mod):
    """Mean simulated / model correlation in bins of fr = |ln(f1/f2)|."""
    fr = np.abs(np.log(centers[:, None] / centers[None, :]))
    iu = np.triu_indices(len(centers), k=1)
    edges = [0.0, 0.1, 0.3, 0.7, 1.5, 3.0]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr[iu] > lo) & (fr[iu] <= hi)
        if m.any():
            rows.append((lo, hi, int(m.sum()), R_mod[iu][m].mean(),
                         R_sim[iu][m].mean(),
                         np.sqrt(np.mean((R_sim[iu][m] - R_mod[iu][m]) ** 2))))
    return rows


def run(n_sim=2000, seed=1, correlation_type='within_site',
        normalizations=('expected', 'per_realization', 'none'),
        apply_to='horizontal', Mw=6.5, f_min=0.2, f_max=24., stride=2,
        sigma=None, output_dir='./verification_output', show=False, dt=0.01):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    print("Verification of interfrequency correlation")
    print(f"  n_sim={n_sim}, seed={seed}, Mw={Mw}, correlation_type={correlation_type}, "
          f"apply_to={apply_to}")

    # --- target: evaluate on nodes of the BA19 grid (no interpolation) -----
    def make_cfg(norm):
        return CorrelatedNoiseConfig(correlation_type=correlation_type, normalization=norm,
                                     apply_to=apply_to, sigma=sigma)
    gen0 = make_cfg(normalizations[0]).build(Mw=Mw)
    nodes = gen0.nodes
    inside = np.flatnonzero((nodes >= f_min) & (nodes <= f_max))[::stride]
    centers = nodes[inside]
    R_model = gen0.R[np.ix_(inside, inside)]
    sd_target = gen0.sigma_nodes[inside]

    # --- baseline (original uncorrelated noise) ---------------------------
    eq0 = build_scenario(n_sim, seed, None, Mw=Mw, dt=dt)
    eas0 = smoothed_eas(eq0.sim_acc, dt, centers)
    R0, sd0, mu0 = sample_statistics(eas0)
    pga0 = pga_geomean(eq0.sim_acc)

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for norm in normalizations:
            eq = build_scenario(n_sim, seed, make_cfg(norm), Mw=Mw, dt=dt)
            eas = smoothed_eas(eq.sim_acc, dt, centers)
            R, sd, mu = sample_statistics(eas)
            results[norm] = dict(eas=eas, R=R, sd=sd, mu=mu, pga=pga_geomean(eq.sim_acc))

    first = normalizations[0]
    Rf = results[first]['R']

    # --- numbers ----------------------------------------------------------
    print("\nCorrelation: simulated vs model, by log-frequency separation fr = |ln(f1/f2)|")
    print(f"  {'fr range':>12s} {'pairs':>6s} {'model':>7s} {'baseline':>9s}"
          + ''.join(f" {n[:9]:>9s}" for n in normalizations) + "   rmse(" + first[:9] + ")")
    rows0 = binned_comparison(centers, R0, R_model)
    rowsn = {n: binned_comparison(centers, results[n]['R'], R_model) for n in normalizations}
    for k, (lo, hi, cnt, mod, base, _) in enumerate(rows0):
        vals = ''.join(f" {rowsn[n][k][4]:9.3f}" for n in normalizations)
        print(f"  {lo:5.2f}-{hi:<5.2f}  {cnt:6d} {mod:7.3f} {base:9.3f}{vals}   {rowsn[first][k][5]:.3f}")
    iu = np.triu_indices(len(centers), k=1)
    print("\n  overall RMSE of the correlation (all pairs, upper triangle):")
    print(f"    baseline        {np.sqrt(np.mean((R0 - R_model)[iu] ** 2)):.3f}")
    for n in normalizations:
        print(f"    {n:<15s} {np.sqrt(np.mean((results[n]['R'] - R_model)[iu] ** 2)):.3f}")
    print(f"  (sampling noise of a correlation with n_sim={n_sim}: ~{1 / np.sqrt(n_sim):.3f})")

    print("\nStandard deviation of ln EAS (target / baseline / simulated), selected frequencies")
    pick = [np.argmin(abs(centers - f)) for f in (0.5, 1., 2., 5., 10., 20.)]
    print(f"  {'f [Hz]':>7s} {'target':>7s} {'baseline':>9s}"
          + ''.join(f" {n[:9]:>9s}" for n in normalizations))
    for j in pick:
        print(f"  {centers[j]:7.2f} {sd_target[j]:7.2f} {sd0[j]:9.2f}"
              + ''.join(f" {results[n]['sd'][j]:9.2f}" for n in normalizations))

    print("\nLevel relative to the baseline (same seed, same scenario)")
    print(f"  {'normalization':>16s} {'median EAS':>11s} {'rms EAS':>9s} {'median PGA':>11s} {'mean PGA':>9s}")
    for n in normalizations:
        r = results[n]
        med = np.exp(np.mean(r['mu'] - mu0))
        rms = np.sqrt(np.mean(r['eas'] ** 2) / np.mean(eas0 ** 2))
        print(f"  {n:>16s} {med:11.2f} {rms:9.2f} "
              f"{np.median(r['pga']) / np.median(pga0):11.2f} {np.mean(r['pga']) / np.mean(pga0):9.2f}")
    print("  (median EAS: geometric mean over realisations and frequencies; rms EAS: "
          "sqrt of mean square over realisations and frequencies)")

    title_type = {'within_site': 'within-site', 'total': 'total (BA19 model)',
                  'between_site': 'between-site'}.get(correlation_type, correlation_type)

    # --- figure 1: Fig. 5-style cross sections ----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (R, lab) in zip(axs, ((Rf, f'simulated with correlated noise [{first}]'),
                                  (R0, 'baseline (original uncorrelated noise)'))):
        for fc, c in zip(COND_FREQS, COND_COLORS):
            j = np.argmin(abs(centers - fc))
            lo, hi = fisher_ci(R[j], n_sim)
            ax.fill_between(centers, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(centers, R[j], '-', color=c, lw=1.4)
            ax.plot(centers, R_model[j], '--', color=c, lw=1.1)
        ax.set_xscale('log'); ax.set_xlim(centers[0], centers[-1]); ax.set_ylim(-0.1, 1.02)
        ax.set_xlabel('F1 (Hz)'); ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
    axs[0].set_ylabel(r'$\rho_\epsilon$')
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'{fc:g} Hz')
               for fc, c in zip(COND_FREQS, COND_COLORS)]
    handles += [plt.Line2D([], [], color='gray', ls='-', label='simulated (95 % band)'),
                plt.Line2D([], [], color='gray', ls='--', label='paper matrix')]
    fig.legend(handles=handles, fontsize=8, ncol=7, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(f'Cross sections of the {title_type} correlation at five conditioning '
                 f'frequencies (as Fig. 5 of Bayless & Abrahamson 2019), Mw {Mw}', fontsize=11)
    f0 = os.path.join(output_dir, 'fig5_style_cross_sections.png')
    fig.savefig(f0, dpi=150, bbox_inches='tight')

    # --- figure 2: contours -----------------------------------------------
    levels = np.linspace(0, 1, 11)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True, sharey=True)
    panels = [(f'Paper matrix ({title_type})', R_model),
              ('Baseline (uncorrelated noise)', R0),
              (f'Simulated, normalization={first}', Rf)]
    for ax, (title, M) in zip(axs[:3], panels):
        cs = ax.contourf(centers, centers, np.clip(M, 0, 1), levels=levels, cmap='viridis')
        ax.set_title(title, fontsize=10)
    cd = axs[3].contourf(centers, centers, Rf - R_model, levels=np.linspace(-0.4, 0.4, 17),
                         cmap='RdBu_r', extend='both')
    axs[3].set_title('Simulated - paper matrix', fontsize=10)
    for ax in axs:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('F1 (Hz)')
    axs[0].set_ylabel('F2 (Hz)')
    fig.colorbar(cs, ax=axs[:3], label='correlation', shrink=0.9)
    fig.colorbar(cd, ax=axs[3], label='difference', shrink=0.9)
    fig.suptitle('Interfrequency correlation of epsilon (smoothed EAS)', fontsize=11)
    f1 = os.path.join(output_dir, 'correlation_contours.png')
    fig.savefig(f1, dpi=150, bbox_inches='tight')

    # --- figure 3: sigma and level ----------------------------------------
    fig2, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(centers, sd_target, 'k--', lw=1.8, label=f'target sigma ({title_type}, Mw {Mw})')
    ax[0].plot(centers, sd0, color='gray', lw=1.2, label='baseline')
    for n in normalizations:
        ax[0].plot(centers, results[n]['sd'], lw=1.2, label=n)
    ax[0].set_xscale('log'); ax[0].set_xlabel('frequency (Hz)')
    ax[0].set_ylabel('std of ln EAS'); ax[0].legend(fontsize=8)
    ax[0].set_title('Standard deviation of ln EAS', fontsize=10)
    for n in normalizations:
        r = results[n]
        ax[1].plot(centers, np.exp(r['mu'] - mu0), lw=1.3, label=n)
        ax[2].plot(centers, np.sqrt(np.mean(r['eas'] ** 2, axis=0) / np.mean(eas0 ** 2, axis=0)),
                   lw=1.3, label=n)
    for a, t in ((ax[1], 'median EAS / baseline median'), (ax[2], 'rms EAS / baseline rms')):
        a.axhline(1, color='k', lw=0.8); a.set_xscale('log')
        a.set_xlabel('frequency (Hz)'); a.set_title(t, fontsize=10); a.legend(fontsize=8)
    fig2.tight_layout()
    f2 = os.path.join(output_dir, 'verification_summary.png')
    fig2.savefig(f2, dpi=150, bbox_inches='tight')
    print(f"\nFigures written to {f0}\n                   {f1}\n                   {f2}")
    if show:
        plt.show()
    plt.close('all')
    return dict(centers=centers, R_model=R_model, R_baseline=R0, results=results,
                sd_baseline=sd0, sd_target=sd_target, figures=(f0, f1, f2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-sim', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mw', type=float, default=6.5)
    p.add_argument('--correlation-type', default='within_site',
                   choices=['within_site', 'total', 'between_site'])
    p.add_argument('--normalization', nargs='+',
                   default=['expected', 'per_realization', 'none'],
                   choices=['expected', 'per_realization', 'none'])
    p.add_argument('--apply-to', default='horizontal', choices=['horizontal', 'all'])
    p.add_argument('--f-min', type=float, default=0.2)
    p.add_argument('--f-max', type=float, default=24.)
    p.add_argument('--stride', type=int, default=2,
                   help='use every k-th node of the 239-frequency BA19 grid')
    p.add_argument('--output-dir', default='./verification_output')
    p.add_argument('--show', action='store_true')
    a = p.parse_args(argv)
    run(n_sim=a.n_sim, seed=a.seed, Mw=a.mw, correlation_type=a.correlation_type,
        normalizations=tuple(a.normalization), apply_to=a.apply_to, f_min=a.f_min,
        f_max=a.f_max, stride=a.stride, output_dir=a.output_dir, show=a.show)


if __name__ == '__main__':
    main()

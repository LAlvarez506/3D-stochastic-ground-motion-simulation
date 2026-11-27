"""
Verification 2 - the ground-motion spectrum and the spectrum scaled by the noise
=================================================================================

Shows, for one point-source scenario, how the deterministic spectrum is turned
into simulated spectra by the noise, with and without the interfrequency
correlation:

  * the deterministic target EAS of the sub-source (from its P-radial, SV-radial
    and SH spectra);
  * the EAS of simulated records (Konno-Ohmachi smoothed, b = 188.5) built with
    the original uncorrelated noise (baseline) and with the correlated noise;
  * the amplitude factor F(f) that the correlated noise applies, with the
    +-1 sigma band of the paper's model;
  * the ensemble median, rms and 16-84 % band against the target, for each
    normalisation, with the value expected from the lognormal model.

Scaling check
-------------
With the corrected noise normalisation in ``scale_spectrum`` the rms EAS of the
baseline records equals the deterministic target at every frequency, whatever
dt and the record length.  The script prints the measured ratio (baseline rms /
target); a value far from 1 means the simulator normalisation depends on dt or
on the record length.

Figures (written to --output-dir)
---------------------------------
spectra_single_realization.png   target, baseline and correlated EAS of one
                                 realisation; amplitude factor F(f) and band.
spectra_ensemble.png             median / rms / 16-84 % bands against the target
                                 for the baseline and each normalisation, and
                                 the ratio to the target with the expected value.
spectra_many_realizations.png    twenty realisations of each, over the target.

Run
---
    python -m stochastic_gm.verification.verify_noise_scaled_spectra \
        --n-sim 1000 --output-dir ./verification_output
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from stochastic_gm import CorrelatedNoiseConfig
from stochastic_gm.verification.verify_interfrequency_correlation import (
    build_scenario, smoothed_eas)

NORM_COLORS = {'expected': '#1f77b4', 'per_realization': '#ff7f0e', 'none': '#2ca02c'}


def target_eas(ps):
    """
    Deterministic horizontal EAS of a point source on its own frequency axis:
    sqrt( (A_sh^2 + A_sv_radial^2 + A_p_radial^2) / 2 ) times the horizontal
    radiation factors H.  (NS^2 + EW^2 = radial^2 + tangential^2 for a
    rotation; P and S contributions are added in energy.)
    """
    Hs, Hp = ps['H_s_ij'], ps['H_p_ij']
    f = np.asarray(ps['frequencies'], dtype=float)
    a2 = (np.asarray(ps['A_sh']) * Hs) ** 2 \
        + (np.asarray(ps['A_sv_radial']) * Hs) ** 2 \
        + (np.abs(np.asarray(ps['A_p_radial'])) * Hp) ** 2
    return f, np.sqrt(0.5 * a2)


def run(n_sim=1000, seed=1, correlation_type='within_site',
        normalizations=('expected', 'per_realization', 'none'),
        apply_to='horizontal', Mw=6.5, f_min=0.2, f_max=24., n_eval=80,
        sigma=None, output_dir='./verification_output', show=False, dt=0.01):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    centers = np.geomspace(f_min, f_max, n_eval)

    def make_cfg(norm):
        return CorrelatedNoiseConfig(correlation_type=correlation_type, normalization=norm,
                                     apply_to=apply_to, sigma=sigma)

    # --- baseline -----------------------------------------------------------
    eq0 = build_scenario(n_sim, seed, None, Mw=Mw, dt=dt)
    ps = eq0.psources[0][0]
    f_t, eas_t = target_eas(ps)
    n_samp = len(eq0.sim_acc[0][0])
    ref = np.interp(centers, f_t, eas_t)
    eas0 = smoothed_eas(eq0.sim_acc, dt, centers)

    # --- correlated ---------------------------------------------------------
    eas = {}
    gens = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for norm in normalizations:
            cfg = make_cfg(norm)
            gens[norm] = cfg.build(Mw=Mw)
            eq = build_scenario(n_sim, seed, cfg, Mw=Mw, dt=dt)
            eas[norm] = smoothed_eas(eq.sim_acc, dt, centers)

    gen = gens[normalizations[0]]
    sig = gen.target_sigma(centers)
    print(f"Scenario: Mw {Mw}, record {n_samp} samples x {dt} s")
    print(f"correlation_type={correlation_type}, apply_to={apply_to}, n_sim={n_sim}")

    # --- numbers ----------------------------------------------------------
    rms0 = np.sqrt(np.mean(eas0 ** 2, axis=0))
    med0 = np.exp(np.mean(np.log(eas0), axis=0))
    pick = [np.argmin(abs(centers - f)) for f in (0.3, 0.5, 1., 2., 5., 10., 20.)]
    print("\nRatio to the target  (1.00 = equal to the deterministic spectrum)")
    hdr = f"  {'f [Hz]':>7s} {'sigma':>6s} {'base rms':>9s} {'base med':>9s}"
    for n in normalizations:
        hdr += f" {n[:9] + ' rms':>14s} {n[:9] + ' med':>14s} {'theory med':>13s}"
    print(hdr)
    expected_med = {'expected': np.exp(-sig ** 2), 'none': np.ones_like(sig),
                    'per_realization': None}
    for j in pick:
        row = f"  {centers[j]:7.2f} {sig[j]:6.2f} {rms0[j] / ref[j]:9.3f} {med0[j] / ref[j]:9.3f}"
        for n in normalizations:
            rms = np.sqrt(np.mean(eas[n][:, j] ** 2)) / ref[j]
            med = np.exp(np.mean(np.log(eas[n][:, j]))) / ref[j]
            em = expected_med[n]
            row += f" {rms:14.3f} {med:14.3f} " + (f"{em[j]:13.3f}" if em is not None else f"{'-':>13s}")
        print(row)
    print("  (theory med: exp(-sigma^2) for 'expected', 1 for 'none'; "
          "'per_realization' has no closed form)")
    print(f"\nBaseline rms / target: mean {np.mean(rms0 / ref):.3f}, "
          f"min {np.min(rms0 / ref):.3f}, max {np.max(rms0 / ref):.3f}")
    if abs(np.mean(rms0 / ref) - 1.0) > 0.1:
        print("  WARNING: the baseline spectrum does not match the deterministic target; "
              "check the noise normalisation in scale_spectrum (dt / record-length dependence).")
    c = 1.0

    # --- figure 1: single realisation ---------------------------------------
    k = 0
    first = normalizations[0]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].loglog(centers, ref, 'k-', lw=2.2, label='deterministic target')
    ax[0].loglog(centers, eas0[k], color='gray', lw=1.3, label='baseline noise')
    for n in normalizations:
        ax[0].loglog(centers, eas[n][k], color=NORM_COLORS[n], lw=1.3,
                     label=f'correlated noise [{n}]')
    ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('smoothed EAS')
    ax[0].set_title('One realisation (same seed, same phases)', fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].semilogx(centers, eas0[k] / ref, color='gray', lw=1.3, label='baseline')
    for n in normalizations:
        ax[1].semilogx(centers, eas[n][k] / ref, color=NORM_COLORS[n], lw=1.3, label=n)
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('EAS / target')
    ax[1].set_title('Same realisation, divided by the target', fontsize=10)
    ax[1].legend(fontsize=8)

    f_plot = np.geomspace(f_min, f_max, 300)
    sg = gen.target_sigma(f_plot)
    for s in range(5):
        F = gen.amplitude_factor(f_plot, seed=seed * 1000 + s)
        ax[2].semilogx(f_plot, F, lw=1.0, alpha=0.8)
    ax[2].semilogx(f_plot, np.exp(-sg ** 2), 'k-', lw=1.8, label='median of F, exp(-sigma^2)')
    ax[2].semilogx(f_plot, np.exp(-sg ** 2 + sg), 'k--', lw=1.2, label='+/- 1 sigma')
    ax[2].semilogx(f_plot, np.exp(-sg ** 2 - sg), 'k--', lw=1.2)
    ax[2].axhline(1, color='gray', lw=0.8)
    ax[2].set_xlabel('frequency (Hz)'); ax[2].set_ylabel('amplitude factor F(f)')
    ax[2].set_title(f"Amplitude factor of five draws [{first}]", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    f_a = os.path.join(output_dir, 'spectra_single_realization.png')
    fig.savefig(f_a, dpi=150, bbox_inches='tight')

    # --- figure 2: ensemble ---------------------------------------------------
    cases = [('baseline noise', eas0, 'gray', None)] + \
            [(f'correlated [{n}]', eas[n], NORM_COLORS[n], n) for n in normalizations]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    axs = axs.ravel()
    for a, (lab, E, col, n) in zip(axs, cases):
        p16, p50, p84 = np.percentile(E, [16, 50, 84], axis=0)
        rms = np.sqrt(np.mean(E ** 2, axis=0))
        a.fill_between(centers, p16, p84, color=col, alpha=0.25, label='16-84 %')
        a.loglog(centers, p50, color=col, lw=1.6, label='median')
        a.loglog(centers, rms, color=col, lw=1.2, ls=':', label='rms')
        a.loglog(centers, ref, 'k-', lw=1.6, label='target')
        a.set_title(lab, fontsize=10); a.set_xlabel('frequency (Hz)')
        a.set_ylabel('smoothed EAS'); a.legend(fontsize=8)
    fig.suptitle(f'Ensemble of {n_sim} realisations against the deterministic spectrum '
                 f'(Mw {Mw}, {correlation_type})', fontsize=11)
    fig.tight_layout()
    f_b = os.path.join(output_dir, 'spectra_ensemble.png')
    fig.savefig(f_b, dpi=150, bbox_inches='tight')

    # --- figure 3: ratio to the target with the expected values -------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].semilogx(centers, med0 / ref, color='gray', lw=1.4, label='baseline')
    ax[1].semilogx(centers, rms0 / ref, color='gray', lw=1.4, label='baseline')
    for n in normalizations:
        med = np.exp(np.mean(np.log(eas[n]), axis=0)) / ref
        rms = np.sqrt(np.mean(eas[n] ** 2, axis=0)) / ref
        ax[0].semilogx(centers, med, color=NORM_COLORS[n], lw=1.4, label=n)
        ax[1].semilogx(centers, rms, color=NORM_COLORS[n], lw=1.4, label=n)
    ax[0].semilogx(centers, np.exp(-sig ** 2), 'k--', lw=1.2, label="expected, 'expected'")
    ax[0].axhline(1, color='k', lw=0.8, ls=':')
    ax[1].axhline(1, color='k', lw=0.8, ls=':')
    ax[0].set_title('Ensemble median EAS / (target)', fontsize=10)
    ax[1].set_title('Ensemble rms EAS / (target)', fontsize=10)
    for a in ax:
        a.set_xlabel('frequency (Hz)'); a.legend(fontsize=8)
    fig.tight_layout()
    f_c = os.path.join(output_dir, 'spectra_ratio_to_target.png')
    fig.savefig(f_c, dpi=150, bbox_inches='tight')

    # --- figure 4: many realisations -----------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for a, (lab, E, col) in zip(axs, (('baseline noise', eas0, 'gray'),
                                      (f'correlated noise [{first}]', eas[first],
                                       NORM_COLORS[first]))):
        for i in range(20):
            a.loglog(centers, E[i], color=col, lw=0.8, alpha=0.7)
        a.loglog(centers, ref, 'k-', lw=2.0, label='target')
        a.set_title(lab, fontsize=10); a.set_xlabel('frequency (Hz)'); a.legend(fontsize=8)
    axs[0].set_ylabel('smoothed EAS')
    fig.tight_layout()
    f_d = os.path.join(output_dir, 'spectra_many_realizations.png')
    fig.savefig(f_d, dpi=150, bbox_inches='tight')

    print(f"\nFigures written to:\n  {f_a}\n  {f_b}\n  {f_c}\n  {f_d}")
    if show:
        plt.show()
    plt.close('all')
    return dict(centers=centers, ref=ref, eas0=eas0, eas=eas, c=c,
                figures=(f_a, f_b, f_c, f_d))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-sim', type=int, default=1000)
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
    p.add_argument('--output-dir', default='./verification_output')
    p.add_argument('--show', action='store_true')
    a = p.parse_args(argv)
    run(n_sim=a.n_sim, seed=a.seed, Mw=a.mw, correlation_type=a.correlation_type,
        normalizations=tuple(a.normalization), apply_to=a.apply_to, f_min=a.f_min,
        f_max=a.f_max, output_dir=a.output_dir, show=a.show)


if __name__ == '__main__':
    main()

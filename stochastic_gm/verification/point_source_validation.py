# -*- coding: utf-8 -*-
"""
stochastic_gm/examples/point_source_validation.py
--------------------------------------------------
Point-source validation example.

Run from any directory once the package is installed:
    python -m stochastic_gm.examples.point_source_validation

Or import directly:
    from stochastic_gm.examples.point_source_validation import run
    run(output_dir="./my_output")

Structure
---------
§1  Setup          – homogeneous half-space, no site effects.
§2  Arrival times  – analytical vs. ray-traced P/S arrival times.
§3  Spectra        – all FAS components on one figure.
§4  Time histories – NS, EW, UD acceleration time histories.
§5  Stress-drop sensitivity – SA (5% damping) across stress drops.

All velocities in km/s, distances in km.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; swap to 'TkAgg' / 'Qt5Agg' for interactive
import matplotlib.pyplot as plt

from stochastic_gm import (
    EQ,
    MediumConfig, SourceConfig, PathConfig,
    SiteConfig, DurationConfig, SimConfig,
)


# ============================================================
# STYLE
# ============================================================
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

SPEC_COLORS = {
    "P vertical":  "#58a6ff",
    "P radial":    "#1f6feb",
    "SV vertical": "#3fb950",
    "SV radial":   "#2ea043",
    "SH":          "#f78166",
}
SD_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor":  TEXT, "xtick.color":    MUTED,
    "ytick.color":      MUTED,"text.color":      TEXT,
    "grid.color":       GRID, "grid.linewidth":  0.6,
    "legend.facecolor": PANEL,"legend.edgecolor": GRID,
    "legend.labelcolor":TEXT,  "font.family":    "monospace",
    "font.size":        9,
})

# ============================================================
# RESPONSE SPECTRUM (Newmark-β, 5% damping)
# ============================================================
def response_spectrum(acc, dt, periods, damping=0.05):
    beta_n, gamma = 0.25, 0.5
    periods = np.array(periods)
    periods = periods[periods > 1e-6]  # avoid division by zero
    
    omega = 2*np.pi / periods
    SA = np.zeros_like(periods)

    for k, w in enumerate(omega):
        c = 2*damping*w
        ks = w**2
        kk = ks + gamma/(beta_n*dt)*c + 1/(beta_n*dt**2)

        n = len(acc)
        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        # Correct initial conditions:
        a[0] = -acc[0]

        for i in range(n-1):
            dp = (-acc[i+1]
                  + (ks + c*gamma/(beta_n*dt)) * u[i]
                  + (1/(beta_n*dt) + c*gamma/beta_n) * v[i]
                  + (1/(2*beta_n) - 1) * a[i])

            du = dp / kk
            da = ((du - u[i]) / (beta_n * dt**2)
                  - v[i] / (beta_n * dt)
                  - (1/(2*beta_n) - 1) * a[i])

            # Correct Newmark velocity update:
            dv = dt * ((1 - gamma) * a[i] + gamma * (a[i] + da))

            u[i+1] = u[i] + du
            v[i+1] = v[i] + dv
            a[i+1] = a[i] + da

        SA[k] = np.max(np.abs(u)) * w**2

    return SA

# ============================================================
# MAIN RUN FUNCTION
# ============================================================
def run(
    # ── Soil ────────────────────────────────────────────────
    Vp:          float = 6.00,   # km/s
    Vs:          float = 3.46,   # km/s
    Rho:         float = 2.70,   # g/cm³
    # ── Geometry ────────────────────────────────────────────
    hypo_depth:  float = 15.0,   # km
    epi_dist:    float = 30.0,   # km  (site is at [epi_dist, 0, 0])
    # ── Source ──────────────────────────────────────────────
    Mw:          float = 6.0,
    stress_drop: float = 50.0,   # bar  (reference run)
    rake:        float = 90.0,
    dip:         float = 45.0,
    strike:      float = 45.0,
    # ── Path ────────────────────────────────────────────────
    Qo:          float = 300.0,
    path_duration: float = 0.05, # s/km
    # ── Simulation ──────────────────────────────────────────
    dt:          float = 0.01,   # s
    seed:        int   = 42,
    # ── Stress-drop sensitivity range ───────────────────────
    stress_drops: list = None,   # bars; None → default range
    # ── Output ──────────────────────────────────────────────
    output_dir:  str   = ".",
    show_plots:  bool  = False,
) -> dict:
    """
    Run the full validation suite and save three figures.

    Returns a dict with keys:
        't_P_analytical', 't_S_analytical',
        't_P_code',       't_S_code',
        'fc_s',           'fc_p',
        'sim_acc'         (NS, EW, UD arrays from the reference run),
        'fig_paths'       (list of saved figure file paths).
    """
    if stress_drops is None:
        stress_drops = [10., 30., 50., 100., 200., 500.]

    os.makedirs(output_dir, exist_ok=True)

    # ── Frequency axis ───────────────────────────────────────
    freq_ps = np.logspace(-1, 2, 500)   # 0.1 – 100 Hz

    # ── Homogeneous half-space ───────────────────────────────
    depth_arr = np.array([0.,  30.])
    vs_arr    = np.array([Vs,  Vs])
    vp_arr    = np.array([Vp,  Vp])
    rho_arr   = np.array([Rho, Rho])

    hypocenter = [0.,      0., hypo_depth]
    site_coord = [epi_dist, 0., 0.]

    amp_freq = np.array([0.01, 200.])
    amp_flat = np.array([1.0,  1.0])

    medium = MediumConfig(Depth=depth_arr, vs=vs_arr, vp=vp_arr, Rho=rho_arr)

    source_ref = SourceConfig(
        stress_drop  = stress_drop,
        rake=rake, dip=dip, strike=strike,
        evolutionary_frequency_model = 'Dang',
    )

    path_cfg = PathConfig(
        Qo=Qo, Q1=0., Q1exp=0.,
        R0=1., R1=100., b1=1., R2=200., b2=0.5, b3=0.,
        use_ray_path = False,
    )

    site_cfg = SiteConfig(amp_freq=amp_freq, amp=amp_flat, kappa=0.04)
    dur_cfg = DurationConfig(path_duration=path_duration)


    def make_sim(seed_=seed):
        return SimConfig(
            dt=dt, freq_ps=freq_ps,
            windows_type='standard',
            epsilon_s=0.2, nu_s=0.05, ftgm_s=0.5,
            epsilon_p=0.2, nu_p=0.05, ftgm_p=0.5,
            gm_type='acc', n_sim=1, seed=seed_,
        )

    # ──────────────────────────────────────────────────────
    # §2  ARRIVAL TIMES
    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STOCHASTIC GM — POINT-SOURCE VALIDATION")
    print("=" * 60)
    print(f"  Vp={Vp} km/s  Vs={Vs} km/s  depth={hypo_depth} km  R_epi={epi_dist} km")

    R_hyp = float(np.linalg.norm(np.array(site_coord) - np.array(hypocenter)))
    t_P_anal = R_hyp / Vp
    t_S_anal = R_hyp / Vs

    print(f"\n§2  ARRIVAL TIMES")
    print(f"  R_hyp = {R_hyp:.3f} km")
    print(f"  Analytical  t_P = {t_P_anal:.4f} s  |  t_S = {t_S_anal:.4f} s")

    # Reference run
    eq_ref = EQ(
        Mw=Mw, medium=medium, source=source_ref, path=path_cfg,
        site_cfg=site_cfg, duration=dur_cfg, sim=make_sim(),
        hypocenter=hypocenter, site_coord=site_coord,
        point_source=True, gm_type='acc', print_info=False,
    )

    ps_out   = eq_ref.theta[0]['PSource'][0]
    t_P_code = ps_out['to_p']
    t_S_code = ps_out['to_s']
    fc_s     = ps_out['fc_s_ij']
    fc_p     = ps_out['fc_p_ij']

    print(f"  Ray-traced  t_P = {t_P_code:.4f} s  |  err = {abs(t_P_code-t_P_anal)*1e3:.2f} ms")
    print(f"  Ray-traced  t_S = {t_S_code:.4f} s  |  err = {abs(t_S_code-t_S_anal)*1e3:.2f} ms")
    print(f"  fc_S = {fc_s:.4f} Hz  |  fc_P = {fc_p:.4f} Hz")

    tol = 0.01
    if abs(t_P_code - t_P_anal) < tol and abs(t_S_code - t_S_anal) < tol:
        print("  ✓  ARRIVAL TIME TEST PASSED")
    else:
        print("  ✗  ARRIVAL TIME TEST FAILED")

    # ──────────────────────────────────────────────────────
    # §3  SPECTRA FIGURE
    # ──────────────────────────────────────────────────────

    # ---------------
    # - Reference
    # --------------
    freqs_ref = [0.00298,0.01445, 0.01908, 0.0272, 0.04322, 0.0865, 0.255,
                 0.427, 0.977, 2.2927, 6.29, 13.39, 27.09, 35.3577, 85.396]
    fas_ref = [0.000028167, 0.3175, 1.0373, 3.0503, 6.5602, 12.65, 31.34, 46.87,
               59.60, 68.28, 57.15, 23.317, 8.586, 2.223, 0.01517]

    # ---------------
    # - Simulation
    # --------------
    print("\n§3  SPECTRA PLOT")
    freqs        = ps_out['frequencies']
    spectra_map  = {
        "P vertical":  ps_out['A_p_vertical'],
        "P radial":    abs(ps_out['A_p_radial']),
        "SV vertical": ps_out['A_sv_vertical'],
        "SV radial":   ps_out['A_sv_radial'],
        "SH":          ps_out['A_sh'],
    }

   
    # ---------------
    # - Spectra
    # --------------
    fig1, ax1 = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax1.set_facecolor(PANEL)
    for label, spec in spectra_map.items():
        ax1.loglog(freqs, spec, lw=1.6, color=SPEC_COLORS[label], label=label)

    ax1.loglog(freqs, spec, lw=1.6, color=SPEC_COLORS[label], label="EXSIM")
    ymin = min(s[s > 0].min() for s in spectra_map.values()) * 0.3
    ax1.axvline(fc_s, color=SPEC_COLORS["SH"],        lw=0.8, ls="--", alpha=0.8)
    ax1.axvline(fc_p, color=SPEC_COLORS["P vertical"], lw=0.8, ls="--", alpha=0.8)

    ax1.text(
        0.05, 0.05,
        f"fc_S = {fc_s:.2f} Hz",
        color=SPEC_COLORS["SH"],
        fontsize=7.5,
        ha="left", va="bottom",
        transform=ax1.transAxes
    )

    ax1.text(
        0.05, 0.16,
        f"fc_P = {fc_p:.2f} Hz",
        color=SPEC_COLORS["P vertical"],
        fontsize=7.5,
        ha="left", va="bottom",
        transform=ax1.transAxes
    )
    ax1.set_xlabel("Frequency  [Hz]")
    ax1.set_ylabel("FAS  [cm/s]")
    ax1.set_title(
        f"FAS Components — Mw {Mw}  |  Δσ {stress_drop} bar  |  "
        f"R_hyp {R_hyp:.1f} km  |  Vp={Vp}  Vs={Vs} km/s",
        fontsize=9, pad=8,
    )
    ax1.legend(
    ncol=2,
    fontsize=8,
    loc="upper left",
    bbox_to_anchor=(0.05, 1.0)
)
    
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.set_xlim([freqs[0], freqs[-1]])
    #plt.tight_layout()
    p1 = os.path.join(output_dir, "fig1_spectra.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {p1}")
    if not show_plots:
        plt.close(fig1)

    # ──────────────────────────────────────────────────────
    # §4  TIME HISTORIES FIGURE
    # ──────────────────────────────────────────────────────
    print("\n§4  TIME HISTORIES")
    NS, EW, UD = eq_ref.sim_acc[0]
    n_s  = len(NS)
    time = np.linspace(0., n_s * dt, n_s)

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True, facecolor=BG)
    fig2.subplots_adjust(hspace=0.08)
    comps2 = [(NS, "NS (North–South)", "#58a6ff"),
              (EW, "EW (East–West)",   "#3fb950"),
              (UD, "UD (Up–Down)",     "#f78166")]
    for ax, (sig, lbl, col) in zip(axes2, comps2):
        ax.set_facecolor(PANEL)
        ax.plot(time, sig, lw=0.7, color=col, alpha=0.9)
        ax.axvline(t_P_code, color="#d2a8ff", lw=0.9, ls="--", alpha=0.8)
        ax.axvline(t_S_code, color="#ffa657", lw=0.9, ls="--", alpha=0.8)
        ax.set_ylabel(f"{lbl}\n[cm/s²]", fontsize=8)
        ax.grid(True, ls="-", alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    yl = axes2[0].get_ylim()
    axes2[0].text(t_P_code + 0.1, yl[1] * 0.70,
                  f"P ({t_P_code:.2f}s)", color="#d2a8ff", fontsize=7.5)
    axes2[0].text(t_S_code + 0.1, yl[1] * 0.70,
                  f"S ({t_S_code:.2f}s)", color="#ffa657", fontsize=7.5)
    axes2[-1].set_xlabel("Time  [s]")
    fig2.suptitle(
        f"Acceleration Time Histories — Mw {Mw}  |  Δσ {stress_drop} bar  |  "
        f"seed={seed}  |  R_hyp {R_hyp:.1f} km",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    p2 = os.path.join(output_dir, "fig2_time_histories.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {p2}")
    if not show_plots:
        plt.close(fig2)

    # ──────────────────────────────────────────────────────
    # §5  STRESS-DROP SENSITIVITY
    # ──────────────────────────────────────────────────────
    print("\n§5  STRESS-DROP SENSITIVITY")
    T_sa  = np.logspace(-2, 1, 80)
    comp_labels = ["NS (North–South)", "EW (East–West)", "UD (Up–Down)"]
    comp_colors = ["#58a6ff", "#3fb950", "#f78166"]

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG)
    fig3.subplots_adjust(wspace=0.28)

    for sd_idx, sd in enumerate(stress_drops):
        col = SD_COLORS[sd_idx % len(SD_COLORS)]
        lw  = 2.0 if sd == stress_drop else 1.0
        al  = 0.95 if sd == stress_drop else 0.65
        print(f"  Δσ = {sd:>5.0f} bar", end="  ", flush=True)

        src_sd = SourceConfig(
            stress_drop=sd, rake=rake, dip=dip, strike=strike,
            evolutionary_frequency_model='Dang',
        )
        eq_sd = EQ(
            Mw=Mw, medium=medium, source=src_sd, path=path_cfg,
            site_cfg=site_cfg, duration=dur_cfg, sim=make_sim(),
            hypocenter=hypocenter, site_coord=site_coord,
            point_source=True, gm_type='acc', print_info=False,
        )
        NS_sd, EW_sd, UD_sd = eq_sd.sim_acc[0]
        for sig, ax3 in zip([NS_sd, EW_sd, UD_sd], axes3):
            SA = response_spectrum(sig, dt, T_sa)
            breakpoint()
            ax3.loglog(T_sa, SA, lw=lw, color=col, alpha=al,
                       label=f"Δσ={sd:.0f} bar")
        print("done")

    for ax3, lbl, cc in zip(axes3, comp_labels, comp_colors):
        ax3.set_facecolor(PANEL)
        ax3.set_xlabel("Period  [s]", fontsize=8)
        ax3.set_ylabel("SA  [cm/s²]",  fontsize=8)
        ax3.set_title(lbl, fontsize=9, color=cc, pad=6)
        ax3.grid(True, which="both", ls="-", alpha=0.4)
        ax3.set_xlim([T_sa[0], T_sa[-1]])
        for sp in ax3.spines.values(): sp.set_edgecolor(GRID)

    # Single de-duplicated legend on last panel
    hdl, lbl_list = axes3[-1].get_legend_handles_labels()
    seen, hu, lu = set(), [], []
    for h, l in zip(hdl, lbl_list):
        if l not in seen:
            seen.add(l); hu.append(h); lu.append(l)
    axes3[-1].legend(hu, lu, fontsize=7.5, loc="lower left")

    fig3.suptitle(
        f"SA (5% damping) — Stress-Drop Sensitivity\n"
        f"Mw {Mw}  |  R_hyp {R_hyp:.1f} km  |  Vp={Vp}  Vs={Vs} km/s  |  "
        f"Reference Δσ={stress_drop} bar (thick line)",
        fontsize=8.5, y=1.02,
    )
    plt.tight_layout()
    p3 = os.path.join(output_dir, "fig3_stress_drop_SA.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {p3}")
    if not show_plots:
        plt.close(fig3)

    print("\n" + "=" * 60)
    print(f"ALL FIGURES IN: {os.path.abspath(output_dir)}")
    print("=" * 60)

    if show_plots:
        plt.show()

    return dict(
        t_P_analytical = t_P_anal,
        t_S_analytical = t_S_anal,
        t_P_code       = t_P_code,
        t_S_code       = t_S_code,
        fc_s           = fc_s,
        fc_p           = fc_p,
        sim_acc        = (NS, EW, UD),
        fig_paths      = [p1, p2, p3],
    )


# ── Allow  `python -m stochastic_gm.examples.point_source_validation` ───────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Point-source validation example")
    parser.add_argument("--output-dir", default="./validation_output",
                        help="Directory for output figures (default: ./validation_output)")
    parser.add_argument("--show",    action="store_true",
                        help="Display figures interactively after saving")
    parser.add_argument("--Vp",      type=float, default=6.00)
    parser.add_argument("--Vs",      type=float, default=3.46)
    parser.add_argument("--depth",   type=float, default=15.0)
    parser.add_argument("--dist",    type=float, default=30.0)
    parser.add_argument("--Mw",      type=float, default=6.0)
    parser.add_argument("--sd",      type=float, default=50.0,
                        help="Reference stress drop [bar]")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    run(
        Vp          = args.Vp,
        Vs          = args.Vs,
        hypo_depth  = args.depth,
        epi_dist    = args.dist,
        Mw          = args.Mw,
        stress_drop = args.sd,
        seed        = args.seed,
        output_dir  = args.output_dir,
        show_plots  = args.show,
    )

# -*- coding: utf-8 -*-
"""
run_finite_fault.py
-------------------
Example script to launch a stochastic ground-motion simulation using
the dict-based finite fault interface of the new EQ class.

The fault is defined as a list of sub-source dictionaries, each carrying
its own centroid, slip, rupture time, and optionally per-patch stress drop,
rake, and dip.  All geometric placement is the user's responsibility —
no internal rotations are applied.

Units
-----
  Depths / distances : km
  Velocities         : km/s
  Stress drop        : bar
  Density            : g/cm³
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

from stochastic_gm.earthquake import EQ
from stochastic_gm.config import (MediumConfig, SourceConfig, PathConfig,
                    SiteConfig, DurationConfig, SimConfig)

# =============================================================================
# 1.  PROPAGATION MEDIUM  (1-D layered half-space)
# =============================================================================
# Depths mark the BOTTOM of each layer [km].
# The first entry of vs/vp corresponds to the surface layer.
medium = MediumConfig(
    system   = 'geographic',          # 'local'  → centroids already in km
                                  # 'geographic' → [lon, lat, depth_km]
    Depth    = np.array([0., 14., 34.]),   # layer-bottom depths [km]
    vs       = np.array([3.563, 3.908, 4.62]), # S-wave velocities   [km/s]
    vp       = np.array([6.2, 6.8, 8.05]), # P-wave velocities   [km/s]
    Rho      = np.array([2.73, 2.87, 3.35]), # densities           [g/cm³]
    velocity_ratio = None,       # only used when vp=None (vp = vs/ratio)
)


# =============================================================================
# 2.  SOURCE PARAMETERS  (SCF model, Dang evolutionary fc)
# =============================================================================
source = SourceConfig(
    # Global fault geometry — used as fallback when a sub-source dict
    # does not provide its own 'rake' or 'dip'.
    dip                        = 60.,    # [deg]
    rake                       = 90.,    # [deg]  pure thrust
    strike                     = None,   # [deg]  (used only for file-based path)

    stress_drop                = 140.,    # global Δσ [bar]
    stress_drop_s              = None,   # if None → uses stress_drop
    stress_drop_p              = None,   # if None → uses stress_drop

    evolutionary_frequency_model = 'Dang',  # 'Motazedian' | 'RIK' | 'Dang'

    # Brune fall-off exponents (gamma=2 → standard Brune)
    gamma_s                    = 2.,
    gamma_p                    = 2.,

    kappa_source               = None,   # source-side kappa filter [s]; None → off
    fcp_multiplier             = None,   # fc_p = alpha/beta * fc_s if None
    F_pulse                    = None,    # pulse fraction [%] — Motazedian model only
    scaling_factor             = 1.,     # global amplitude multiplier
)

# =============================================================================
# 3.  PATH / ATTENUATION
# =============================================================================
path = PathConfig(
    # Geometrical spreading — piecewise power law
    # Z(R) = (R0/R)^b1              for R  ≤ R1
    #        (R0/R1)^b1*(R1/R)^b2   for R1 < R ≤ R2
    #        ... * (R2/R)^b3        for R  > R2
    R0 = 1.,     # reference distance [km]
    R1 = 70.,    # first transition   [km]
    R2 = 130.,   # second transition  [km]
    b1 = 1.0,    # exponent ≤ R1
    b2 = 0.5,    # exponent R1–R2
    b3 = 0.5,    # exponent > R2

    # Anelastic attenuation  Q(f) = max(Qo, Q1 * f^Q1exp)
    Qo     = 1e-8,
    Q1     = 126.,
    Q1exp  = 1.05,
    Qp_factor = 'udias',   # 'udias' → Q_p = (3/4)*(alpha/beta)^2 * Q_s

    # Non-parametric attenuation table (GIT) — None → use parametric Q above
    Att_non_par   = None,
    Att_non_par_s = None,
    Att_non_par_p = None,

    # Distance metric for attenuation and duration
    use_ray_path = False,   # False → hypocentral distance (default)
                             # True  → ray-traced R_s / R_p
)

# =============================================================================
# 4.  SITE
# =============================================================================
site_cfg = SiteConfig(
    # kappa [s] — high-frequency site attenuation
    kappa   = 0.03,
    kappa_s = None,   # if None → uses kappa
    kappa_p = None,   # if None → uses kappa

    # fk — corner for the Butterworth site filter; None → off
    fk   = None,
    fk_s = None,
    fk_p = None,

    # f_max hard cut-off [Hz]; None → off
    f_max     = None,
    f_max_exp = 4.,

    # site_lowpass — Butterworth order; None → off
    site_lowpass = None,

    # Velocity-based amplification curve
    amp_freq = None,   # frequency array [Hz] for amp table; None → no amp
    amp      = None,   # amplification values (same length as amp_freq)

    # GIT transfer functions (amplitude-only, applied inside PointSource)
    TF_GIT   = None,
    TF_GIT_s = None,
    TF_GIT_p = None,

    # Fixed incidence angle override [deg]; None → ray-traced
    incidence_manual = None,
)

# =============================================================================
# 5.  DURATION
# =============================================================================
duration = DurationConfig(
    # Path-duration slope [s/km]  (T_path = path_duration × R)
    path_duration     = 0.05,

    boolean_tgm_path   = True,  # True → use fixed Tgm_path instead of slope
    Tgm_path           = 40.,

    boolean_tgm_source = True,  # True → use fixed Tgm_source
    Tgm_source         = 10.,     # or 'courboulex_sub' / 'courboulex_other'

    # Per-sub-source fixed durations (only used when ps_specific_duration=True)
    ps_specific_duration = False,
    duration_swave       = None,
    duration_pwave       = None,

    duration_max = None,         # hard cap on waveform length [s]; None → off
)

# =============================================================================
# 6.  SIMULATION SETTINGS
# =============================================================================
sim = SimConfig(
    dt           = 0.01,                            # time step [s]
    freq_ps      = np.linspace(0., 100., 1001),     # frequency array for FAS [Hz]
    n_sim        = 1,                               # number of stochastic realisations
    n_processors = multiprocessing.cpu_count() - 1, # parallel workers (1 = sequential)
    noise_stddev = 1.,                              # white-noise standard deviation
    seed         = 42,                              # reproducibility; None → random

    # Window type: 'standard' (parametric exponential) or 'GIT' (empirical)
    windows_type = 'standard',

    # Parametric exponential window parameters (used when windows_type='standard')
    # P-wave window
    epsilon_p = 0.2,
    nu_p      = 0.05,
    ftgm_p    = 2.0,
    # S-wave window
    epsilon_s = 0.2,
    nu_s      = 0.05,
    ftgm_s    = 2.0,

    # GIT window arrays (only used when windows_type='GIT')
    window_standard_time = None,
    window_prms     = None,   window_prms_dur = None,
    window_srms     = None,   window_srms_dur = None,
    window_p3       = None,   window_p3_dur   = None,
    window_s3       = None,   window_s3_dur   = None,

    T_pads   = False,    # True → add zero pads = 7.5/fc_s_ij before window
    sh_only  = False,    # True → output SH component only
)

# =============================================================================
# 7.  HYPOCENTER AND SITE  (local coordinates [x, y, z] in km)
# =============================================================================
hypocenter = [35.497, 34.07, 6.36]    # [x_km, y_km, depth_km]
site_coord = [35.53, 33.89,  0.]   # observation point at surface

 
# =============================================================================
# 8.  LOAD FINITE FAULT FROM CSV
# =============================================================================
csv_path = 'lebanon_faille.csv'           # ← adjust path as needed
df = pd.read_csv(csv_path, sep=';')
 
# Filter out patches with zero slip — they contribute nothing
df = df[df['slip'] > 0.].reset_index(drop=True)
 
print(f"Loaded {len(df)} active sub-sources from '{csv_path}'")
 
# Build the list of sub-source dicts.
# - depth is converted from metres to km.
# - dip and rake are taken per-patch from the CSV.
# - strike is not passed: the centroids are already geographically placed,
#   so no rotation is needed inside the tool.
# - width and length are ignored.
 
subsource_list = []
for _, row in df.iterrows():
    subsource_list.append({
        'centroid' : [row['lon'],
                      row['lat'],
                      row['depth'] / 1000.],   # m → km
        'slip'     :  row['slip'],              # m
        'rupt_t'   :  row['rupture'],           # s
        'dip'      :  row['dip'],               # deg — per-patch override
        'rake'     :  row['rake'],              # deg — per-patch override
        # 'sd'     :  50.,  # uncomment to set per-patch stress drop [bar]
    })
 
# Quick sanity check
depths_km = [ss['centroid'][2] for ss in subsource_list]
print(f"  Depth range  : {min(depths_km):.2f} – {max(depths_km):.2f} km")
print(f"  Slip range   : {df['slip'].min():.3f} – {df['slip'].max():.3f} m")
print(f"  Rupt_t range : {df['rupture'].min():.2f} – {df['rupture'].max():.2f} s")

# =============================================================================
# 9.  RUN THE SIMULATION
# =============================================================================
sim_result = EQ(
    Mw            = 7.3,
    medium        = medium,
    source        = source,
    path          = path,
    site_cfg      = site_cfg,
    duration      = duration,
    sim           = sim,
    hypocenter    = hypocenter,
    site_coord    = site_coord,
    point_source  = False,        # False → finite fault mode
    subsource_list = subsource_list,
    source_file   = None,
    gm_type       = 'acc',
    print_info    = True,
)

# =============================================================================
# 10.  INSPECT RESULTS
# =============================================================================
# sim_result.sim_acc is a list of length n_sim.
# Each element is a tuple (NS, EW, UD) of numpy arrays.

dt = sim.dt
n_sim = sim.n_sim

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
labels = ['N–S', 'E–W', 'U–D']

for i_sim in range(n_sim):
    NS, EW, UD = sim_result.sim_acc[i_sim]
    t = np.arange(len(NS)) * dt
    for ax, comp, label in zip(axes, [NS, EW, UD], labels):
        ax.plot(t, comp * 100., alpha=0.5, lw=0.8,
                label=f'sim {i_sim+1}' if label == 'N–S' else '_')

for ax, label in zip(axes, labels):
    ax.set_ylabel(f'{label}  [cm/s²]')
    ax.grid(True, lw=0.4)

axes[0].legend(fontsize=8, ncol=n_sim)
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'Stochastic finite-fault simulation  |  Mw 6.5  |  '
             f'{len(subsource_list)} sub-sources  |  R_hyp ≈ {np.linalg.norm(np.array(site_coord)-np.array(hypocenter)):.1f} km',
             fontsize=11)
plt.tight_layout()
plt.savefig('finite_fault_waveforms.png', dpi=150)
plt.show()

print("\nSimulation complete.")
print(f"  Realisations : {n_sim}")
print(f"  Sub-sources  : {len(subsource_list)}")
print(f"  R_hypocentral: {np.linalg.norm(np.array(site_coord)-np.array(hypocenter)):.2f} km")
print(f"  Waveform length (sim 0, NS): {len(sim_result.sim_acc[0][0]) * dt:.2f} s")

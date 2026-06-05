# run_simulation.py

import numpy as np
from config import MediumConfig, SourceConfig, PathConfig, SiteConfig, DurationConfig, SimConfig
from earthquake import EQ

# ------------------------------------------------------------------ #
# 1. MEDIUM  — 1-D velocity model
# ------------------------------------------------------------------ #
medium = MediumConfig(
    Depth = np.array([0., 2000., 5000., 15000., 30000., 60000.]),  # m
    vs    = np.array([0.5,  1.5,   2.5,   3.2,   3.6,   4.0  ]) * 1000.,  # m/s
    vp    = np.array([1.0,  2.8,   4.5,   5.8,   6.5,   7.2  ]) * 1000.,  # m/s
    Rho   = np.array([1.8,  2.1,   2.4,   2.6,   2.7,   2.9  ]),           # g/cm³
    system = 'local',   # coordinates already in metres
)

# ------------------------------------------------------------------ #
# 2. SOURCE
# ------------------------------------------------------------------ #
source = SourceConfig(
    stress_drop  = 50.,     # bar
    rake         = 90.,     # degrees  (pure thrust)
    dip          = 45.,     # degrees
    strike       = 200.,    # degrees
    source_type  = 'SCF',   # single corner frequency
    evolutionary_frequency_model = 'Dang',
)

# ------------------------------------------------------------------ #
# 3. PATH  — piecewise geometric spreading + Q
# ------------------------------------------------------------------ #
path = PathConfig(
    Qo    = 200.,   Q1  = 0.,    Q1exp = 0.,   # Q(f) = 200 (frequency-independent)
    R0    = 1.,                                  # m  (reference distance)
    R1    = 80000., b1  = 1.0,                  # body-wave spreading up to 80 km
    R2    = 200000.,b2  = 0.5,   b3    = 0.0,  # transition + far field
)

# ------------------------------------------------------------------ #
# 4. SITE  — flat amplification + kappa filter
# ------------------------------------------------------------------ #
site = SiteConfig(
    amp_freq = np.array([0.1, 1., 5., 10., 25., 50., 100.]),   # Hz
    amp      = np.array([1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0]),
    kappa    = 0.04,   # s
)

# ------------------------------------------------------------------ #
# 5. DURATION
# ------------------------------------------------------------------ #
duration = DurationConfig(
    path_duration      = 0.05,    # s/m  (Tgm increases 0.05 s per metre of path)
    boolean_tgm_source = False,   # use frequency-based source duration
    boolean_tgm_path   = False,   # use distance-based path duration
)

# ------------------------------------------------------------------ #
# 6. SIMULATION SETTINGS
# ------------------------------------------------------------------ #
sim = SimConfig(
    dt           = 0.01,    # s
    n_sim        = 5,       # number of stochastic realisations
    gm_type      = 'acc',   # output: acceleration
    windows_type = 'standard',
    # Saragoni-Hart window shape
    ftgm_p = 0.85,  epsilon_p = 0.2,  nu_p = 0.05,
    ftgm_s = 0.85,  epsilon_s = 0.2,  nu_s = 0.05,
    seed   = 42,
)

# ------------------------------------------------------------------ #
# 7. RUN
# ------------------------------------------------------------------ #
eq = EQ(
    Mw          = 6.8,
    medium      = medium,
    source      = source,
    path        = path,
    site        = site,
    duration    = duration,
    sim         = sim,
    hypocenter  = [0.,     0.,     5.],   # x, y, depth [km]
    site_coord  = [2., 0.,     0.    ],   # x, y, z [km]
    point_source = True,
)

# ------------------------------------------------------------------ #
# 8. ACCESS RESULTS
# ------------------------------------------------------------------ #
for i, (NS, EW, UD) in enumerate(eq.accelerations):
    t = np.arange(len(NS)) * sim.dt
    print(f'Sim {i+1}: PGA_NS={max(abs(NS)):.4f}  PGA_EW={max(abs(EW)):.4f}')
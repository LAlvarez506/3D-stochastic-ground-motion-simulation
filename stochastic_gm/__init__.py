# -*- coding: utf-8 -*-
"""
stochastic_gm
=============
Stochastic ground-motion simulator based on the Boore (2003) stochastic
method extended to multi-component (P, SV, SH) finite-fault and
point-source models.

Supported source model
----------------------
SCF (single-corner-frequency, Brune 1970) with three sub-source
corner-frequency evolution rules:
    * 'Motazedian'  – Motazedian & Atkinson (2005)
    * 'RIK'         – fc from local sub-source stress drop
    * 'Dang'        – Dang et al. (2021) moment-ratio scaling

Quick start
-----------
>>> from stochastic_gm import EQ
>>> from stochastic_gm import MediumConfig, SourceConfig, PathConfig
>>> from stochastic_gm import SiteConfig, DurationConfig, SimConfig
>>> import numpy as np
>>>
>>> eq = EQ(
...     Mw         = 6.0,
...     medium     = MediumConfig(
...                      Depth = np.array([0., 30.]),
...                      vs    = np.array([3.46, 3.46]),
...                      vp    = np.array([6.00, 6.00]),
...                      Rho   = np.array([2.7,  2.7]),
...                  ),
...     source     = SourceConfig(stress_drop=50., rake=90., dip=45., strike=45.),
...     path       = PathConfig(Qo=300., Q1=0., Q1exp=0.,
...                             R0=1., R1=100., b1=1., R2=200., b2=0.5, b3=0.),
...     site_cfg   = SiteConfig(amp_freq=np.array([0.01, 200.]),
...                             amp     =np.array([1.0,  1.0])),
...     duration   = DurationConfig(path_duration=0.05),
...     sim        = SimConfig(dt=0.01, n_sim=1, seed=42,
...                            epsilon_s=0.2, nu_s=0.05, ftgm_s=0.5,
...                            epsilon_p=0.2, nu_p=0.05, ftgm_p=0.5),
...     hypocenter = [0.,  0.,  15.],
...     site_coord = [30., 0.,  0.],
... )
>>> NS, EW, UD = eq.sim_acc[0]

Unit conventions
----------------
All velocities  : km/s
All distances   : km
Seismic moment  : dyne·cm
Stress drop     : bar
Frequency       : Hz
"""

# ── Core simulation classes ──────────────────────────────────────────────────
from .earthquake    import EQ
from .point_source  import PointSource

# ── Typed configuration dataclasses ─────────────────────────────────────────
from .config import (
    MediumConfig,
    SourceConfig,
    PathConfig,
    SiteConfig,
    DurationConfig,
    SimConfig,
)

# ── Physics / utility functions (available for advanced users) ───────────────
from .physics import (
    # geometry
    vector_plane_angle,
    locate_site,
    rupture_distances,
    find_nearest,
    # wave propagation
    snells_direct_prop,
    ray_propagation_bisection,
    FS_pwaves,
    FS_swaves,
    density_profile,
    # signal processing
    white_noise,
    FFT,
    IFFT,
    norm_power_spec,
    scale_spectrum,
    # site
    site_amplification,
    site_attenuation,
    # window
    window_function,
)

__all__ = [
    # classes
    "EQ",
    "PointSource",
    # configs
    "MediumConfig",
    "SourceConfig",
    "PathConfig",
    "SiteConfig",
    "DurationConfig",
    "SimConfig",
    # physics
    "vector_plane_angle",
    "locate_site",
    "rupture_distances",
    "find_nearest",
    "snells_direct_prop",
    "ray_propagation_bisection",
    "FS_pwaves",
    "FS_swaves",
    "density_profile",
    "white_noise",
    "FFT",
    "IFFT",
    "norm_power_spec",
    "scale_spectrum",
    "site_amplification",
    "site_attenuation",
    "window_function",
]

__version__ = "0.1.0"

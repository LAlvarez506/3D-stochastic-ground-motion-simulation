# -*- coding: utf-8 -*-
"""
config.py
---------
Typed input configuration for the stochastic ground-motion simulator.

Replace the flat ``parameters`` dict previously passed to EQ.__init__ with
structured dataclass instances.  Each dataclass carries sensible defaults
so that only the parameters relevant to a given run need to be specified.

Supported source types
----------------------
Only the **SCF** (single-corner-frequency) source model is supported.
Corner-frequency evolution per sub-source is controlled by
``evolutionary_frequency_model``:
    ``'Motazedian'``  – Motazedian & Atkinson (2005) pulse-fraction scaling.
    ``'RIK'``         – fc derived from local sub-source stress drop.
    ``'Dang'``        – Dang et al. (2021) moment-ratio scaling.

Usage example
-------------
    from config import MediumConfig, SourceConfig, PathConfig, SiteConfig, DurationConfig, SimConfig

    eq = EQ(
        Mw        = 6.8,
        medium    = MediumConfig(Depth=depth_arr, vs=vs_arr, vp=vp_arr, Rho=rho_arr),
        source    = SourceConfig(stress_drop=50., rake=90., dip=45., strike=200.),
        path      = PathConfig(Qo=200., Q1=0., Q1exp=0.,
                               R0=1., R1=80., b1=1.,
                               R2=200., b2=0.5, b3=0.),
        site_cfg  = SiteConfig(amp_freq=freq_arr, amp=amp_arr, kappa=0.04),
        duration  = DurationConfig(path_duration=0.05),
        sim       = SimConfig(dt=0.01, n_sim=10, seed=42),
        hypocenter = [0., 0., 15.],   # km  (local Cartesian)
        site_coord = [20., 0., 0.],   # km
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import random


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------

@dataclass
class MediumConfig:
    """
    Description of the 1-D propagation medium.

    Parameters
    ----------
    Depth : array-like
        Layer interface depths [km].  May start at 0 (surface included) or
        at the first interface below the surface; the constructor normalises
        either convention automatically.
    vs : array-like
        S-wave velocity profile [km/s].  One entry per layer.
    vp : array-like, optional
        P-wave velocity profile [km/s].  If None, derived from vs /
        velocity_ratio.
    Rho : array-like, optional
        Density profile [g/cm³].  If None, estimated from Boore (2003).
        NOTE: density_profile() expects vs in km/s.
    velocity_ratio : float, optional
        vs / vp ratio, used only when vp is None.
    system : str
        Coordinate system used to express hypocenter and site positions.
        ``'local'``      – Cartesian [x, y, z_depth] in kilometres,
                           coordinates are used as-is.
        ``'geographic'`` – [longitude, latitude, depth_km]; the EQ class
                           converts them to local Cartesian automatically.
    """
    Depth:          np.ndarray
    vs:             np.ndarray
    vp:             Optional[np.ndarray] = None
    Rho:            Optional[np.ndarray] = None
    velocity_ratio: Optional[float]      = None
    system:         str                  = 'local'

    def __post_init__(self):
        self.Depth = np.asarray(self.Depth, dtype=float)
        self.vs    = np.asarray(self.vs,    dtype=float)
        if self.vp  is not None: self.vp  = np.asarray(self.vp,  dtype=float)
        if self.Rho is not None: self.Rho = np.asarray(self.Rho, dtype=float)

        # Normalise depth array: insert surface (0.) if absent
        if self.Depth[0] != 0.:
            expected = len(self.vs) - 1
            if len(self.Depth) != expected:
                raise ValueError(
                    f"When surface is not included in Depth, len(Depth) must equal "
                    f"len(vs) - 1  (got {len(self.Depth)} and {len(self.vs)})."
                )
            self.Depth = np.insert(self.Depth, 0, 0.)
        else:
            if len(self.Depth) != len(self.vs):
                raise ValueError(
                    f"When surface is included in Depth, len(Depth) must equal "
                    f"len(vs)  (got {len(self.Depth)} and {len(self.vs)})."
                )


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """
    Earthquake source parameters.

    Only the SCF (single-corner-frequency) source model is supported.

    Parameters
    ----------
    stress_drop : float
        Target stress drop [bar].
    rake : float
        Rake angle [degrees].
    dip : float
        Dip angle [degrees].  A value of 0 is replaced by 0.0001 internally.
    strike : float
        Strike angle [degrees].  A value of 0 is replaced by 0.0001 internally.
    evolutionary_frequency_model : str
        Sub-source fc scaling rule.  One of ``'Motazedian'``, ``'RIK'``,
        ``'Dang'``.
    kappa_source : float, optional
        Source-side kappa filter [s].
    F_pulse : float, optional
        Maximum pulse fraction for Motazedian model [%].
    stress_drop_s : float, optional
        Separate S-wave stress drop [bar].  Overrides stress_drop for S.
    stress_drop_p : float, optional
        Separate P-wave stress drop [bar].  Overrides stress_drop for P.
    gamma_s, gamma_p : float
        High-frequency decay exponent of the source spectrum for S and P waves.
    fcp_multiplier : float, optional
        Manual multiplier applied to fc_s to obtain fc_p.  If None the ratio
        alpha/beta is used.
    depth_ref : str, optional
        Reference for depth normalisation, e.g. ``'hypocenter'`` or
        ``'lower_depth'``.
    lower_depth : float, optional
        Lower depth of the fault [km].
    random_strike : bool
        If True, strike is drawn uniformly from (0, 360].
    random_dip : bool
        If True, dip is drawn uniformly from (0, 90].
    scaling_factor : float
        Global amplitude scaling factor applied to all spectra.
    """
    stress_drop:    float
    rake:           float
    dip:            float
    strike:         float
    evolutionary_frequency_model: str = 'Dang'
    kappa_source:   Optional[float] = None
    F_pulse:        Optional[float] = None
    stress_drop_s:  Optional[float] = None
    stress_drop_p:  Optional[float] = None
    gamma_s:        float = 2.
    gamma_p:        float = 2.
    fcp_multiplier: Optional[float] = None
    depth_ref:      Optional[str]   = None
    lower_depth:    Optional[float] = None
    random_strike:  bool  = False
    random_dip:     bool  = False
    scaling_factor: float = 1.

    def __post_init__(self):
        valid_evo = ('Motazedian', 'RIK', 'Dang')
        if self.evolutionary_frequency_model not in valid_evo:
            raise ValueError(
                f"evolutionary_frequency_model must be one of {valid_evo}; "
                f"got '{self.evolutionary_frequency_model}'."
            )

        if self.random_strike:
            self.strike = random.uniform(0.0001, 360.)
        elif self.strike == 0.:
            self.strike = 0.0001

        if self.random_dip:
            self.dip = random.uniform(0.0001, 90.)
        elif self.dip == 0.:
            self.dip = 0.0001


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """
    Path propagation parameters (geometrical spreading + anelastic attenuation).

    Piecewise geometrical spreading uses three segments:
        R ≤ R1           : G(R) = (R0/R)^b1
        R1 < R ≤ R2      : G(R) = (R0/R1)^b1 * (R1/R)^b2
        R > R2           : G(R) = (R0/R1)^b1 * (R1/R2)^b2 * (R2/R)^b3

    Q model: Q(f) = max(Qo, Q1 * f^Q1exp)

    Distance used for path attenuation
    ------------------------------------
    By default the **hypocentral distance** is used for both P and S waves
    (i.e. the straight-line source-to-site distance, identical for both
    wave types).  Set ``use_ray_path=True`` to replace this with the true
    ray path length computed by the ray-tracing bisection method, which
    differs between P and S waves because they travel at different velocities
    through the layered medium.

    Parameters
    ----------
    Qo, Q1, Q1exp : float
        Quality-factor model coefficients.
    R0, R1, b1, R2, b2, b3 : float
        Geometrical spreading segment parameters [km].
    geometric_spreading : str
        Label for the spreading model (informational).
    Qp_factor : str or float
        Scaling of Q for P-waves relative to S-waves.
        ``'udias'`` uses (3/4)*(alpha/beta)^2; a float is used directly.
    use_ray_path : bool
        If True, use the ray-traced path length R_s / R_p instead of the
        hypocentral distance.  Default False (hypocentral).
    Att_non_par, Att_non_par_s, Att_non_par_p : dict, optional
        Non-parametric attenuation lookup tables from GIT inversion.
        Each dict must contain keys ``'Att'``, ``'freqs'``, ``'R'``.
    """
    Qo:    float
    Q1:    float
    Q1exp: float
    R0:    float
    R1:    float
    b1:    float
    R2:    float
    b2:    float
    b3:    float
    geometric_spreading: str          = 'simplified'
    Qp_factor:           object       = 'udias'   # str or float
    use_ray_path:        bool         = False
    Att_non_par:         Optional[dict]  = None
    Att_non_par_s:       Optional[dict]  = None
    Att_non_par_p:       Optional[dict]  = None


# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------

@dataclass
class SiteConfig:
    """
    Site amplification and attenuation parameters.

    The site filter G(f) = Amp(f) * D(f) where Amp is an interpolated
    amplification function and D is one of: kappa, f_max, or lowpass.

    Parameters
    ----------
    amp_freq : array-like
        Frequency axis for the amplification function [Hz].
    amp : array-like
        Amplification values at amp_freq (linear, not dB).
    kappa : float, optional
        Broadband site kappa [s].
    kappa_s, kappa_p : float, optional
        Wave-type-specific kappa [s].  Override kappa for S and P respectively.
    fk : float or 'corner', optional
        Frequency at which kappa attenuation begins [Hz].
        Use ``'corner'`` to set it equal to the corner frequency.
        NOTE: if a ``ValueError`` is raised mentioning that exactly one of
        kappa / f_max / lowpass must be provided, verify that ``fk`` is
        either a positive float or the string ``'corner'``, not an unexpected
        type.
    fk_s, fk_p : float, optional
        Wave-type-specific fk values [Hz].
    f_max : float, optional
        f_max diminution parameter [Hz].
    f_max_exp : float
        Exponent in the f_max filter.
    site_lowpass : dict, optional
        Butterworth low-pass description with keys ``'fmax'`` and ``'order'``.
    TF_GIT : dict, optional
        GIT-derived amplitude-only transfer function applied at spectrum
        construction time (see PointSource._apply_git_tf).  All components
        share the same TF.
    TF_GIT_s, TF_GIT_p : dict, optional
        Wave-type-specific amplitude-only GIT transfer functions.
    incidence_manual : float, optional
        Override the computed incidence angle with a fixed value [degrees].
    """
    amp_freq:       np.ndarray
    amp:            np.ndarray
    kappa:          Optional[float]            = None
    kappa_s:        Optional[float]            = None
    kappa_p:        Optional[float]            = None
    fk:             Optional[Union[float, str]] = None   # float or 'corner'
    fk_s:           Optional[float]            = None
    fk_p:           Optional[float]            = None
    f_max:          Optional[float]            = None
    f_max_exp:      float                      = 8.
    site_lowpass:   Optional[dict]             = None
    TF_GIT:         Optional[dict]             = None
    TF_GIT_s:       Optional[dict]             = None
    TF_GIT_p:       Optional[dict]             = None
    incidence_manual: Optional[float]          = None

    def __post_init__(self):
        self.amp_freq = np.asarray(self.amp_freq, dtype=float)
        self.amp      = np.asarray(self.amp,      dtype=float)

        if self.fk is not None:
            if not (isinstance(self.fk, (int, float)) or self.fk == 'corner'):
                raise ValueError(
                    "SiteConfig.fk must be a positive float or the string 'corner'; "
                    f"got {self.fk!r}.  If site_attenuation raises a 'provide exactly "
                    "one of kappa/f_max/lowpass' error, check that fk is set correctly."
                )


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

@dataclass
class DurationConfig:
    """
    Ground-motion duration parameters.

    The total duration per sub-source is:  Tgm = T_source + T_path

    Parameters
    ----------
    path_duration : float, optional
        Slope [s/km] for the path-duration model T_path = path_duration * R.
        Required when boolean_tgm_path is False.
    boolean_tgm_path : bool
        If True, use the fixed value Tgm_path instead of the distance-based model.
    Tgm_path : float
        Fixed path duration [s], used when boolean_tgm_path is True.
    boolean_tgm_source : bool
        If True, use the fixed value Tgm_source instead of the magnitude-based model.
    Tgm_source : float or str
        Fixed source duration [s] when boolean_tgm_source is True.
        Alternatively, pass ``'courboulex_sub'`` or ``'courboulex_other'`` to
        sample randomly from the Courboulex et al. empirical model.
        Note: the Mo-dependent median is then the reference for the lognormal
        draw (mu = log(T), sigma = s_ln), not the mean.
    ps_specific_duration : bool
        If True, use fixed duration_pwave / duration_swave for all sub-sources.
    duration_swave, duration_pwave : float
        Fixed durations [s] for S and P waves when ps_specific_duration is True.
    duration_max : float, optional
        Hard cap on total signal duration [s].
    """
    path_duration:         Optional[float] = None
    boolean_tgm_path:      bool  = False
    Tgm_path:              float = 0.
    boolean_tgm_source:    bool  = False
    Tgm_source:            object = 0.    # float or str
    ps_specific_duration:  bool  = False
    duration_swave:        float = 0.
    duration_pwave:        float = 0.
    duration_max:          Optional[float] = None
    use_afshari   : bool  = False   # True → compute Tgm from AS2016
    vs30_afshari  : float = 760.    # Vs30 for AS2016 site term [m/s]
    z1pt0_afshari : float = None    # z1.0 [m]; None → estimated from Vs30


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Simulation control parameters.

    Parameters
    ----------
    dt : float
        Time step [s].
    freq_ps : array-like
        Frequency axis for spectrum computation [Hz].
    windows_type : str
        Window type: ``'GIT'`` for empirical GIT windows, ``'standard'`` for
        parametric Saragoni-Hart windows.
    noise_stddev : float
        Standard deviation of the white-noise excitation.
    T_pads : bool
        If True, add zero-padding equal to 7.5/fc_s to avoid spectral truncation.
    gm_type : str
        Output ground-motion type: ``'acc'``, ``'vel'``, or ``'disp'``.
    n_sim : int
        Number of independent realisations.
    n_processors : int
        Number of parallel processors (currently sequential; kept for future use).
    seed : int, optional
        Master random seed.  If None, a random seed is drawn.
    sh_only : bool
        If True, output only the SH component (EXSIM-style).

    Window parameters (standard Saragoni-Hart shape)
    -------------------------------------------------
    ftgm_p, ftgm_s : float
        Fraction of Tgm at which the window peak occurs.
    epsilon_p, epsilon_s : float
        Window shape parameter ε (time of peak / total duration).
    nu_p, nu_s : float
        Window shape parameter ν (decay rate).

    GIT empirical window tables
    ---------------------------
    window_prms, window_srms, window_p3, window_s3 : array-like
        Amplitude arrays of the four GIT window components, defined on
        window_standard_time.
    window_prms_dur, window_srms_dur, window_p3_dur, window_s3_dur : float
        Duration of each GIT window component [s].
    window_standard_time : array-like
        Time axis shared by all four GIT window amplitude arrays [s].
    """
    dt:           float
    freq_ps:      np.ndarray = field(default_factory=lambda: np.linspace(0., 100., 1001))
    windows_type: str   = 'standard'
    noise_stddev: float = 1.
    T_pads:       bool  = False
    gm_type:      str   = 'acc'
    n_sim:        int   = 1
    n_processors: int   = 1
    seed:         Optional[int] = None
    sh_only:      bool  = False

    # Standard window parameters
    ftgm_p:    float = 0.
    epsilon_p: float = 0.
    nu_p:      float = 0.
    ftgm_s:    float = 0.
    epsilon_s: float = 0.
    nu_s:      float = 0.

    # GIT window tables
    window_prms:     Optional[np.ndarray] = None
    window_srms:     Optional[np.ndarray] = None
    window_p3:       Optional[np.ndarray] = None
    window_s3:       Optional[np.ndarray] = None
    window_prms_dur: Optional[float]      = None
    window_srms_dur: Optional[float]      = None
    window_p3_dur:   Optional[float]      = None
    window_s3_dur:   Optional[float]      = None
    window_standard_time: Optional[np.ndarray] = None

    def __post_init__(self):
        self.freq_ps = np.asarray(self.freq_ps, dtype=float)

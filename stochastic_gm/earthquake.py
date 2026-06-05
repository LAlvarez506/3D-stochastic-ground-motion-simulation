# -*- coding: utf-8 -*-
"""
earthquake.py
-------------
EQ class: orchestrates the stochastic ground-motion simulation for a
finite-fault or point-source model.

Supported source model
----------------------
Only the SCF (single-corner-frequency) source model is supported.
Sub-source corner-frequency evolution: 'Motazedian', 'RIK', or 'Dang'.

Velocity / distance units
--------------------------
All velocities are in **km/s**.
All depths and distances (hypocenter, site coordinates, R_hypocentral,
R_epicentral, R0, R1, R2) are in **km**.

Bug fixes vs. original EQ_niigata.py
--------------------------------------
* ``kappa_s`` was assigned from ``kappa_p`` — fixed to ``kappa_s``.
* S-wave window used P-wave parameters — fixed.
* ``Tgm_source is 'courboulex_sub'`` used identity comparison ``is``
  — changed to ``==``.
* Lognormal sampling: mu is now ``log(T)`` (sampling around the median T).
* All class-level attributes replaced with instance attributes.
* Input via typed dataclasses (config.py) instead of a flat dict.
* JA19 / JA19_2S source models removed; only SCF is supported.
* Path-attenuation distance controlled by PathConfig.use_ray_path boolean.
"""

from __future__ import annotations
import math
import time
import random
import multiprocessing as mp

import numpy as np
import scipy.interpolate

from .config import (MediumConfig, SourceConfig, PathConfig,
                    SiteConfig, DurationConfig, SimConfig)
from .point_source import PointSource
from .physics import (locate_site, rupture_distances, find_nearest,
                     white_noise, FFT, IFFT, norm_power_spec,
                     scale_spectrum, window_function)

# GSource is an external dependency kept as-is


def _process_one_subsource(args: tuple) -> dict:
    """
    Module-level wrapper — processes noise convolution for a single sub-source.
    Boore/SMSIM-consistent implementation.
    """
    import numpy as np
    import math
    from stochastic_gm.physics import white_noise, FFT, scale_spectrum

    ps, sim_seed, dt, standard_freqs, duration_max, _ = args

    Normalization = None
    n_samples     = ps['n_samples']

    theta_p       = ps['incidence_p']
    theta_s       = ps['incidence_s']
    H_p           = ps['H_p_ij']
    H_s           = ps['H_s_ij']
    atr           = {}

    # ------------------------------------------------------------------
    # 1. Generate unit-variance white noise (NOT duration-scaled)
    # ------------------------------------------------------------------
    wNoise = white_noise(n_samples, standard_dev=1.0, seed=sim_seed)

    # ------------------------------------------------------------------
    # Window construction helper
    # ------------------------------------------------------------------
    def _make_win(key, t0_key):
        wn      = ps['window'][key]['wind']
        start   = int((ps[t0_key] + ps['t_rupture']) / dt)
        pad_end = n_samples - (len(wn) + start)
        win     = np.concatenate((np.zeros(start), wn,
                                  np.zeros(max(pad_end, 0))))
        if duration_max is not None:
            cap = int(duration_max / dt)
            if cap < len(win):
                win = win[:cap]
        return win

    # ------------------------------------------------------------------
    # FFT helper with **time-domain energy normalization**
    # ------------------------------------------------------------------
    def _fft_std(win):
        wn = wNoise * win

        # --- RMS normalization ---
        E = np.sum(wn**2) * dt
        if E <= 0.0:
            raise RuntimeError("Noise energy is zero after windowing.")
        wn /= math.sqrt(E)
        wn *= math.sqrt(n_samples * dt)

        f_, fr = FFT(wn, dt, norm=Normalization)
        return f_, fr

    # ------------------------------------------------------------------
    # GIT window case
    # ------------------------------------------------------------------
    if ps['window_type'] == 'GIT':

        def _make_win_git(key, t0_key):
            wn      = ps['window'][key]['wind']
            start   = int((ps[t0_key] + ps['t_rupture']) / dt)
            pad_end = n_samples - (len(wn) + start)
            win     = np.concatenate((np.zeros(start), wn,
                                      np.zeros(max(pad_end, 0))))
            if duration_max is not None:
                cap = int(duration_max / dt)
                if cap < len(win):
                    win = win[:cap]
            return win

        PRMS_win = _make_win_git('prms', 'to_p')
        SRMS_win = _make_win_git('srms', 'to_s')
        P3_win   = _make_win_git('p3',   'to_p')
        S3_win   = _make_win_git('s3',   'to_s')

        time_arr = np.arange(n_samples) * dt

        fft_prms, fr_prms = _fft_std(PRMS_win)
        fft_srms, fr_srms = _fft_std(SRMS_win)
        fft_p3,   fr_p3   = _fft_std(P3_win)
        fft_s3,   fr_s3   = _fft_std(S3_win)

        atr['P_vertical'],  atr['t_acc_pv']  = scale_spectrum(
            ps['A_p_vertical'], fft_p3, fr_p3, time_arr,
            standard_freqs, factor=H_p,
            incidence=theta_p, polarization='vertical', wave_type='P'
        )
        atr['P_radial'],    atr['t_acc_pr']  = scale_spectrum(
            ps['A_p_radial'], fft_prms, fr_prms, time_arr,
            standard_freqs, factor=H_p,
            incidence=theta_p, polarization='radial', wave_type='P'
        )
        atr['SV_vertical'], atr['t_acc_svv'] = scale_spectrum(
            ps['A_sv_vertical'], fft_s3, fr_s3, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='vertical', wave_type='SV'
        )
        atr['SV_radial'],   atr['t_acc_svr'] = scale_spectrum(
            ps['A_sv_radial'], fft_srms, fr_srms, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='radial', wave_type='SV'
        )
        atr['SH'],          atr['t_acc_sh']  = scale_spectrum(
            ps['A_sh'], fft_srms, fr_srms, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='tangential', wave_type='SH'
        )

    # ------------------------------------------------------------------
    # Standard window case
    # ------------------------------------------------------------------
    else:
        P_win = _make_win('P', 'to_p')
        S_win = _make_win('S', 'to_s')

        time_arr = (
            np.arange(int(duration_max / dt)) * dt
            if duration_max is not None and int(duration_max / dt) < n_samples
            else np.arange(n_samples) * dt
        )

        fft_p, fr_p = _fft_std(P_win)
        fft_s, fr_s = _fft_std(S_win)

        atr['P_vertical'],  atr['t_acc_pv']  = scale_spectrum(
            ps['A_p_vertical'], fft_p, fr_p, time_arr,
            standard_freqs, factor=H_p,
            incidence=theta_p, polarization='vertical', wave_type='P'
        )
        atr['P_radial'],    atr['t_acc_pr']  = scale_spectrum(
            ps['A_p_radial'], fft_p, fr_p, time_arr,
            standard_freqs, factor=H_p,
            incidence=theta_p, polarization='radial', wave_type='P'
        )
        atr['SV_vertical'], atr['t_acc_svv'] = scale_spectrum(
            ps['A_sv_vertical'], fft_s, fr_s, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='vertical', wave_type='SV'
        )
        atr['SV_radial'],   atr['t_acc_svr'] = scale_spectrum(
            ps['A_sv_radial'], fft_s, fr_s, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='radial', wave_type='SV'
        )
        atr['SH'],          atr['t_acc_sh']  = scale_spectrum(
            ps['A_sh'], fft_s, fr_s, time_arr,
            standard_freqs, factor=H_s,
            incidence=theta_s, polarization='tangential', wave_type='SH'
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    atr['phi']       = ps['phi']
    atr['dt']        = dt
    atr['n_samples'] = (
        int(min(duration_max / dt, n_samples))
        if duration_max is not None else n_samples
    )

    return atr

# def _process_one_subsource(args: tuple) -> dict:
#     """
#     Module-level wrapper — processes noise convolution for a single sub-source.
#     Required at module level for multiprocessing pickling.
#     """
#     import numpy as np
#     import math
#     from stochastic_gm.physics import white_noise, FFT, norm_power_spec, scale_spectrum

#     ps, sim_seed, dt, standard_freqs, duration_max, stddev = args

#     Normalization = None
#     n_samples     = ps['n_samples']
    
#     theta_p       = ps['incidence_p']
#     theta_s       = ps['incidence_s']
#     H_p           = ps['H_p_ij']
#     H_s           = ps['H_s_ij']
#     atr           = {}

#     T = n_samples * dt
#     sigma = math.sqrt(1.0 / T)
#     wNoise        = white_noise(n_samples, standard_dev=sigma, seed=sim_seed)

#     print(f"DEBUG H_s={H_s:.4e}, H_p={H_p:.4e}", flush=True)
#     print(f"DEBUG n_samples={n_samples}, dt={dt}", flush=True)
#     print(f"DEBUG max(A_sh)={np.max(ps['A_sh']):.4e}", flush=True)

#     def _make_win(key, t0_key):
#         wn      = ps['window'][key]['wind']
#         start   = int((ps[t0_key] + ps['t_rupture']) / dt)
#         pad_end = n_samples - (len(wn) + start)
#         win     = np.concatenate((np.zeros(start), wn,
#                                   np.zeros(max(pad_end, 0))))
#         if duration_max is not None:
#             cap = int(duration_max / dt)
#             if cap < len(win):
#                 win = win[:cap]
#         return win

#     def _fft_std(win):
#         wn_    = wNoise * win
#         fac    = norm_power_spec(wn_, dt)
#         f_, fr = FFT(wn_, dt, norm=Normalization)
#         return f_ * fac, fr

#     if ps['window_type'] == 'GIT':
#         def _make_win_git(key, t0_key):
#             wn      = ps['window'][key]['wind']
#             start   = int((ps[t0_key] + ps['t_rupture']) / dt)
#             pad_end = n_samples - (len(wn) + start)
#             win     = np.concatenate((np.zeros(start), wn,
#                                       np.zeros(max(pad_end, 0))))
#             if duration_max is not None:
#                 cap = int(duration_max / dt)
#                 if cap < len(win):
#                     win = win[:cap]
#             return win

#         PRMS_win = _make_win_git('prms', 'to_p')
#         SRMS_win = _make_win_git('srms', 'to_s')
#         P3_win   = _make_win_git('p3',   'to_p')
#         S3_win   = _make_win_git('s3',   'to_s')
#         time_arr = np.linspace(0., n_samples * dt, n_samples)

#         fft_prms, fr_prms = _fft_std(PRMS_win)
#         fft_srms, fr_srms = _fft_std(SRMS_win)
#         fft_p3,   fr_p3   = _fft_std(P3_win)
#         fft_s3,   fr_s3   = _fft_std(S3_win)

#         atr['P_vertical'],  atr['t_acc_pv']  = scale_spectrum(ps['A_p_vertical'],  fft_p3,   fr_p3,   time_arr, standard_freqs, factor=H_p, incidence=theta_p, norm=Normalization, polarization='vertical',   wave_type='P')
#         atr['P_radial'],    atr['t_acc_pr']  = scale_spectrum(ps['A_p_radial'],    fft_prms, fr_prms, time_arr, standard_freqs, factor=H_p, incidence=theta_p, norm=Normalization, polarization='radial',     wave_type='P')
#         atr['SV_vertical'], atr['t_acc_svv'] = scale_spectrum(ps['A_sv_vertical'], fft_s3,   fr_s3,   time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='vertical',   wave_type='SV')
#         atr['SV_radial'],   atr['t_acc_svr'] = scale_spectrum(ps['A_sv_radial'],   fft_srms, fr_srms, time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='radial',     wave_type='SV')
#         atr['SH'],          atr['t_acc_sh']  = scale_spectrum(ps['A_sh'],          fft_srms, fr_srms, time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='tangential', wave_type='SH')

#     else:
#         P_win = _make_win('P', 'to_p')
#         S_win = _make_win('S', 'to_s')

#         print(f"DEBUG max(wNoise)={np.max(np.abs(wNoise)):.4e}", flush=True)
#         print(f"DEBUG max(S_win)={np.max(np.abs(S_win)):.4e}", flush=True)

#         if duration_max is not None and int(duration_max / dt) < n_samples:
#             time_arr = np.linspace(0., duration_max, int(duration_max / dt))
#         else:
#             time_arr = np.linspace(0., n_samples * dt, n_samples)

#         fft_p, fr_p = _fft_std(P_win)
#         fft_s, fr_s = _fft_std(S_win)

#         print(f"DEBUG max(fft_s)={np.max(np.abs(fft_s)):.4e}", flush=True)

#         atr['P_vertical'],  atr['t_acc_pv']  = scale_spectrum(ps['A_p_vertical'],  fft_p, fr_p, time_arr, standard_freqs, factor=H_p, incidence=theta_p, norm=Normalization, polarization='vertical',   wave_type='P')
#         atr['P_radial'],    atr['t_acc_pr']  = scale_spectrum(ps['A_p_radial'],    fft_p, fr_p, time_arr, standard_freqs, factor=H_p, incidence=theta_p, norm=Normalization, polarization='radial',     wave_type='P')
#         atr['SV_vertical'], atr['t_acc_svv'] = scale_spectrum(ps['A_sv_vertical'], fft_s, fr_s, time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='vertical',   wave_type='SV')
#         atr['SV_radial'],   atr['t_acc_svr'] = scale_spectrum(ps['A_sv_radial'],   fft_s, fr_s, time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='radial',     wave_type='SV')
#         atr['SH'],          atr['t_acc_sh']  = scale_spectrum(ps['A_sh'],          fft_s, fr_s, time_arr, standard_freqs, factor=H_s, incidence=theta_s, norm=Normalization, polarization='tangential', wave_type='SH')
#         atr['windowed_p'] = wNoise * P_win
#         atr['windowed_s'] = wNoise * S_win

#         print(f"DEBUG max(SH time-domain)={np.max(np.abs(atr['SH'])):.4e}", flush=True)

#     atr['phi']       = ps['phi']
#     atr['dt']        = dt
#     atr['n_samples'] = (int(min(duration_max / dt, n_samples))
#                         if duration_max is not None else n_samples)
#     return atr


def _build_point_source(sett: dict) -> dict:
    """Module-level wrapper for PointSource construction — required for multiprocessing."""
    return PointSource(sett).ps_info()[0]

class EQ:
    """
    Stochastic ground-motion simulator (SCF source model).

    Parameters
    ----------
    Mw : float
        Moment magnitude.
    medium : MediumConfig
        1-D propagation medium. Depths in km, velocities in km/s.
    source : SourceConfig
        Source parameters (SCF model only).
    path : PathConfig
        Path / attenuation parameters. Distances in km.
    site_cfg : SiteConfig
    duration : DurationConfig
    sim : SimConfig
    hypocenter : list [x, y, z]  km  (local) or [lon, lat, depth_km] (geographic)
    site_coord : list [x, y, z]  km  (local) or [lon, lat, depth_km] (geographic)
    point_source : bool
        True  → single-point-source model (hypocenter = source centroid).
        False → finite-fault model; requires *subsource_list*.
    subsource_list : list of dict
        Subsource descriptions for finite-fault mode (one dict per sub-source).
    gm_type : str
        ``'acc'``, ``'vel'``, or ``'disp'``.
    print_info : bool, optional
        Verbose output toggle.
    """

    def __init__(self,
                 Mw:          float,
                 medium:      MediumConfig,
                 source:      SourceConfig,
                 path:        PathConfig,
                 site_cfg:    SiteConfig,
                 duration:    DurationConfig,
                 sim:         SimConfig,
                 hypocenter:  list,
                 site_coord:  list,
                 point_source: bool = True,
                 subsource_list: list = None,
                 gm_type:     str   = 'acc',
                 print_info:  bool  = False):

        t0 = time.time()
        self.print_info   = print_info
        self.Mw           = Mw
        self.Mo           = 10. ** ((Mw + 10.7) * 1.5)
        self.gm_type      = gm_type
        self.point_source   = point_source
        self.subsource_list = subsource_list

        # Store config objects
        self.medium   = medium
        self.src_cfg  = source
        self.path_cfg = path
        self.site_cfg = site_cfg
        self.dur_cfg  = duration
        self.sim_cfg  = sim

        # Set random seed
        # Use SeedSequence + spawn to guarantee independent streams per sub-source
        ss = np.random.SeedSequence(sim.seed)
        self._seed_seq = ss
        np.random.seed(ss.entropy % (2**32) if sim.seed is not None else None)

        # Compute seismic moment and source duration if empirical model requested
        self.Tgm_source = self._resolve_tgm_source(duration.Tgm_source)

        # Coordinate conversion
        self.local_coords = (medium.system == 'local')
        self.hypocenter_geographic = hypocenter   # ← add this line, before _set_coordinates()
        self._set_coordinates(hypocenter, site_coord)

        # Common params dict passed to add_noise
        self.params = {
            'dt':             sim.dt,
            'sim_seed':       [],
            'PSource':        None,
            'standard_freqs': sim.freq_ps,
            'incidence_manual': site_cfg.incidence_manual,
            'duration_max':   duration.duration_max,
            'stddev':         sim.noise_stddev,
        }

        # Run
        self.run_simulation()

        if self.print_info:
            print(f"EQ simulation completed in {time.time() - t0:.2f} s")

    # ------------------------------------------------------------------
    # Coordinate setup
    # ------------------------------------------------------------------
    def _set_coordinates(self, hypocenter, site_coord):
        if self.local_coords:
            self.hypocenter = hypocenter   # [x km, y km, z km]
            self.site       = site_coord   # [x km, y km, z km]
        else:
            self.hypocenter = [0., 0., hypocenter[2]]
            self.site       = locate_site(hypocenter, site_coord,
                                          reference=[0., 0.], depth=True)

    # ------------------------------------------------------------------
    # Empirical source-duration model
    # ------------------------------------------------------------------
    def _resolve_tgm_source(self, tgm_source) -> float:
        """
        Resolve the Tgm_source parameter.

        FIX: original used ``is`` for string comparison (always False);
        changed to ``==``.
        FIX: lognormal mu is now ``log(T)`` (median-preserving).
        """
        if isinstance(tgm_source, str):
            Mo_Nm = self.Mo * 1e-7   # dyne-cm → N·m
            if tgm_source == 'courboulex_sub':
                T    = 10. ** (0.28 * np.log10(Mo_Nm) - 4.32)
                s_ln = 0.32
            elif tgm_source == 'courboulex_other':
                T    = 10. ** (0.31 * np.log10(Mo_Nm) - 4.90)
                s_ln = 0.34
            else:
                raise ValueError(f"Unknown Tgm_source model: '{tgm_source}'")
            # FIX: sample from lognormal with mu = log(T) (preserves median T)
            mu_normal    = np.log(T)
            sigma_normal = s_ln
            return float(np.random.lognormal(mean=mu_normal, sigma=sigma_normal))
        else:
            return float(tgm_source)

    # ------------------------------------------------------------------
    # Side computations (corner frequencies, distances)
    # ------------------------------------------------------------------
    def _side_computations(self):
        """
        Compute event-level corner frequencies and rupture distances.

        All velocities are in km/s; distances in km.

        Corner-frequency formula (Boore 2003 / Brune 1970, SMSIM convention):
            fc [Hz] = 4.9E6 * beta [km/s] * (Δσ [bar] / M0 [dyne·cm])^(1/3)

        Reference: Boore (2003, Pure Appl. Geophys.), relation
            Δσ · M0 · fc = 0.4906 · β³
        with β in km/s, M0 in dyne·cm, Δσ in bars.
        """
        depth = self.medium.Depth   # km
        vs    = self.medium.vs      # km/s
        vp    = self.medium.vp      # km/s  (or None)

        idx   = np.where(depth >= self.Hypocenter[2])[0][0]
        beta  = vs[idx - 1]          # km/s at source layer
        alpha = (vp[idx - 1] if vp is not None
                 else beta / self.medium.velocity_ratio)  # km/s

        sd_s = (self.src_cfg.stress_drop_s
                if self.src_cfg.stress_drop_s is not None
                else self.src_cfg.stress_drop)
        sd_p = (self.src_cfg.stress_drop_p
                if self.src_cfg.stress_drop_p is not None
                else self.src_cfg.stress_drop)

        # SCF corner frequencies [Hz]  — Boore (2003) SMSIM convention:
        #   fc = 0.4906 * beta [km/s] * (Δσ [bar] / M0 [dyne·cm])^(1/3)
        self.fc_s = 4.9e6 * beta  * (sd_s / self.Mo) ** (1. / 3.)
        self.fc_p = (alpha * self.fc_s / beta
                     if self.src_cfg.stress_drop_p is None
                     else 4.9e6 * alpha * (sd_p / self.Mo) ** (1. / 3.))

        # Distances [km]
        self.R_hypocentral = np.linalg.norm(
            np.array(self.site) - np.array(self.Hypocenter))
        self.R_epicentral  = np.linalg.norm(
            np.array(self.site) - np.array([self.Hypocenter[0],
                                             self.Hypocenter[1], 0.]))
        self.R_rup, self.R_jb, self.R_x = rupture_distances(
            self.ps_centroid, self.site)

        #print(f"DEBUG fc_s={self.fc_s:.4f} Hz, fc_p={self.fc_p:.4f} Hz")
        #print(f"DEBUG beta={beta:.3f} km/s, Mo={self.Mo:.3e} dyne·cm")
        #print(f"DEBUG stress_drop={sd_s:.1f} bar")


    # ------------------------------------------------------------------
    # Settings dict passed to PointSource
    # ------------------------------------------------------------------
    def _build_sett(self) -> dict:
        """Assemble the settings dict consumed by PointSource.__init__."""
        src = self.src_cfg
        pth = self.path_cfg
        ste = self.site_cfg
        dur = self.dur_cfg
        sim = self.sim_cfg
        med = self.medium

        return {
            # Source
            'Mo_ij':          self.Mo,
            'activation':     1,
            'Mo':             self.Mo,
            'Moave':          self.Mo,
            'N_psources':     1,
            'dip':            src.dip,
            'rake':           src.rake,
            'stress_drop_target': src.stress_drop,
            'stress_drop':    src.stress_drop,
            'stress_drop_p':  src.stress_drop_p,
            'stress_drop_s':  src.stress_drop_s,
            'gamma_p':        src.gamma_p,
            'gamma_s':        src.gamma_s,
            'F_pulse':        src.F_pulse,
            'kappa_source':   src.kappa_source,
            'fcp_multiplier': src.fcp_multiplier,
            'evolutionary_frequency_model': src.evolutionary_frequency_model,
            'fc_p':           None,  # filled in by run_simulation after _side_computations
            'fc_s':           None,
            'scaling_factor': src.scaling_factor,
            't_rupture':      0.,
            'ps_centroid':    None,
            'site_coord':     None,
            # Medium — depths [km], velocities [km/s]
            'depth':          med.Depth,
            'vs':             med.vs,
            'vp':             med.vp,
            'rho':            med.Rho,
            'velocity_ratio': med.velocity_ratio,
            # Path — distances [km]
            'R0':             pth.R0,
            'R1':             pth.R1,
            'b1':             pth.b1,
            'R2':             pth.R2,
            'b2':             pth.b2,
            'b3':             pth.b3,
            'use_ray_path':   pth.use_ray_path,
            'Qo':             pth.Qo,
            'Q1':             pth.Q1,
            'Q1exp':          pth.Q1exp,
            'Qp_factor':      pth.Qp_factor,
            'Att_non_par':    pth.Att_non_par,
            'Att_non_par_s':  pth.Att_non_par_s,
            'Att_non_par_p':  pth.Att_non_par_p,
            # Site
            'amp_freq':       ste.amp_freq,
            'amp':            ste.amp,
            'kappa':          ste.kappa,
            'kappa_s':        ste.kappa_s,
            'kappa_p':        ste.kappa_p,
            'fk':             ste.fk,
            'fk_s':           ste.fk_s,
            'fk_p':           ste.fk_p,
            'f_max':          ste.f_max,
            'f_max_exp':      ste.f_max_exp,
            'site_lowpass':   ste.site_lowpass,
            'TF_GIT':         ste.TF_GIT,
            'TF_GIT_s':       ste.TF_GIT_s,
            'TF_GIT_p':       ste.TF_GIT_p,
            'incidence_manual': ste.incidence_manual,
            # Duration
            'boolean_tgm_source':  dur.boolean_tgm_source,
            'Tgm_source':          self.Tgm_source,
            'boolean_tgm_path':    dur.boolean_tgm_path,
            'Tgm_path':            dur.Tgm_path,
            'path_duration':       dur.path_duration,
            'ps_specific_duration': dur.ps_specific_duration,
            'duration_pwave':      dur.duration_pwave,
            'duration_swave':      dur.duration_swave,
            # Simulation
            'freq_ps':        sim.freq_ps,
            'dt':             sim.dt,
            'gm_type':        self.gm_type,
        }

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    def run_simulation(self):
        sett           = self._build_sett()
        idealizations  = []
        n_sources_list = []

        if self.point_source:
            self.Hypocenter  = self.hypocenter
            self.ps_centroid = self.Hypocenter
            self._side_computations()

            # ── Afshari duration BEFORE PointSource construction ──
            if self.dur_cfg.use_afshari:
                from .physics import afshari_stewart_2016
                Mw_ps = self.Mw   # point source uses full event magnitude
                D595, _, _, _ = afshari_stewart_2016(
                    Mw   = Mw_ps,
                    Rrup = self.R_hypocentral,
                    rake = self.src_cfg.rake,
                    vs30 = self.dur_cfg.vs30_afshari,
                    z1pt0= self.dur_cfg.z1pt0_afshari,
                    imt  = 'rsd595'
                )
                print(f"  Afshari D5-95 : {D595:.2f} s "
                    f"(R={self.R_hypocentral:.1f} km)", flush=True)
                sett['boolean_tgm_source'] = True
                sett['Tgm_source']         = D595
                sett['boolean_tgm_path']   = True
                sett['Tgm_path']           = 0.
            # ──────────────────────────────────────────────────────

            sett['ps_centroid'] = self.ps_centroid
            sett['site_coord']  = self.site
            sett['fc_p']        = self.fc_p
            sett['fc_s']        = self.fc_s
            sett['stress_drop'] = self.src_cfg.stress_drop
            sett['Moave']       = self.Mo
            self.R_emax         = self.R_hypocentral

            ps = PointSource(sett)
            idealizations.append(ps.ps_info())
            n_sources_list.append(1)

        elif self.subsource_list is not None:
            
            self.Hypocenter = self.hypocenter

            ps_centroid, ff_mo, ff_sd, ff_activation, ff_rupture_time, ff_rake, ff_dip = \
                self.set_geometry_from_dict(self.subsource_list)

            print(f"\n[EQ] Finite fault — dict input", flush=True)
            print(f"  Sub-sources loaded : {len(self.subsource_list)}", flush=True)
            print(f"  Active patches     : {int(np.sum(ff_mo > 0.))}", flush=True)
            print(f"  Total Mo           : {self.Mo:.3e} dyne·cm", flush=True)
            print(f"  Mw                 : {self.Mw}", flush=True)
            print(f"  Processors         : {self.sim_cfg.n_processors}", flush=True)
            
            self.ps_centroid = ps_centroid
            self._side_computations()

            sett['site_coord'] = self.site
            sett['fc_p']       = self.fc_p
            sett['fc_s']       = self.fc_s
            sett['Moave']      = float(np.mean(ff_mo[ff_mo > 0.]))

            psource_i = self._mesh(ps_centroid, ff_sd, ff_mo,
                                ff_activation, ff_rupture_time, sett,
                                ff_rake=ff_rake, ff_dip=ff_dip)
            idealizations.append(psource_i)
            n_sources_list.append(len(psource_i))
        


        # Generate noise realisations and aggregate
        # -------------------------------------------------------
        # - Run the simulation
        # -------------------------------------------------------
        self.sim_acc = []
        for psource, N_ps in zip(idealizations, n_sources_list):

            print(f"\n[EQ] Starting noise convolution — "
                f"{self.sim_cfg.n_sim} realisations, "
                f"{N_ps} sub-sources each ...", flush=True)

            for i_sim in range(self.sim_cfg.n_sim):
                p            = self.params.copy()
                child_seeds  = self._seed_seq.spawn(N_ps)
                seeds        = [int(cs.generate_state(1)[0]) for cs in child_seeds]
                windows, dur = self._build_windows(psource, N_ps)
                n_samples    = int(dur / self.sim_cfg.dt)
                p['PSource'] = self._pack_point_sources(psource, windows, n_samples, N_ps)
                p['sim_seed'] = seeds

                # Process immediately — never accumulate all n_sim in memory
                point = self.add_noise(p)
                acc   = self.aggregate_signal(point)
                self.sim_acc.append(acc)

                # Free immediately
                del point, p
                print(f"  [Convolution] Realisation {i_sim+1}/"
                    f"{self.sim_cfg.n_sim} done.", flush=True)

            print(f"[EQ] Convolution complete — "
                f"{len(self.sim_acc)} realisations in total.\n", flush=True)
            
    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _build_windows(self, psource: list, N_ps: int):
        """Return (Windows dict, total duration [s])."""
        sim  = self.sim_cfg
        dur  = self.dur_cfg
        Windows  = {}
        duration = 0.

        for j in range(N_ps):
            if sim.windows_type == 'GIT':
                temp    = {}
                keys_w  = ['prms', 'srms', 'p3', 's3']
                wfuncs  = [sim.window_prms,    sim.window_srms,
                           sim.window_p3,      sim.window_s3]
                wdurs   = [sim.window_prms_dur, sim.window_srms_dur,
                           sim.window_p3_dur,  sim.window_s3_dur]
                x = np.array(sim.window_standard_time)
                for key, wf, wd in zip(keys_w, wfuncs, wdurs):
                    fn   = scipy.interpolate.interp1d(x, wf, kind='linear')
                    t_w  = np.arange(0., x.max(), sim.dt)
                    temp[key] = {'time': t_w, 'wind': fn(t_w)}
                Windows[j]  = temp
                T_pads      = 7.5 / psource[j]['fc_s_ij'] if sim.T_pads else 0.
                total_time  = max(
                    psource[j]['to_s'] + psource[j]['t_rupture'] + wdurs[-1] + T_pads,
                    psource[j]['to_p'] + psource[j]['t_rupture'] + wdurs[-1] + T_pads,
                )

            else:
                wp = {'epsilon_window': sim.epsilon_p, 'nu_window': sim.nu_p,
                      'f_tgm': sim.ftgm_p}
                ws = {'epsilon_window': sim.epsilon_s, 'nu_window': sim.nu_s,
                      'f_tgm': sim.ftgm_s}
                time_p, wind_p = window_function(wp, psource[j]['tgm_p'], sim.dt,
                                                 decay_value=0.001)
                time_s, wind_s = window_function(ws, psource[j]['tgm_s'], sim.dt,
                                                 decay_value=0.001)

                if len(time_p) == 0 or len(time_s) == 0:
                    raise RuntimeError(
                        f"window_function returned an empty array for sub-source {j}. "
                        "Check that tgm_p / tgm_s are positive (verify path_duration, "
                        "stress_drop, and distance units are consistent)."
                    )

                Windows[j]  = {'P': {'time': time_p, 'wind': wind_p},
                                'S': {'time': time_s, 'wind': wind_s}}
                T_pads      = 7.5 / psource[j]['fc_s_ij'] if sim.T_pads else 0.
                total_time  = max(
                    psource[j]['to_s'] + psource[j]['t_rupture'] + float(time_s.max()) + T_pads,
                    psource[j]['to_p'] + psource[j]['t_rupture'] + float(time_p.max()) + T_pads,
                )

            duration = max(duration, total_time)

        duration = round(duration, 2) + 1.0
        return Windows, duration

    def _pack_point_sources(self, psource: list, Windows: dict,
                             n_samples: int, N_ps: int) -> list:
        """Pack psource data into the format expected by add_noise."""
        out = []
        for k in range(N_ps):
            ps = {
                'Mo_ij':         psource[k]['Mo_ij'],
                'frequencies':   psource[k]['frequencies'],
                'A_p_vertical':  psource[k]['A_p_vertical'],
                'A_p_radial':    psource[k]['A_p_radial'],
                'A_sv_vertical': psource[k]['A_sv_vertical'],
                'A_sv_radial':   psource[k]['A_sv_radial'],
                'A_sh':          psource[k]['A_sh'],
                'window':        Windows[k],
                'window_type':   self.sim_cfg.windows_type,
                'to_p':          psource[k]['to_p'],
                'to_s':          psource[k]['to_s'],
                't_rupture':     psource[k]['t_rupture'],
                'n_samples':     n_samples,
                'fc_p_ij':       psource[k]['fc_p_ij'],
                'fc_s_ij':       psource[k]['fc_s_ij'],
                'H_p_ij':        psource[k]['H_p_ij'],
                'H_s_ij':        psource[k]['H_s_ij'],
                'phi':           psource[k]['azimuth'],
                'incidence_s':   psource[k]['incidence_s'],
                'incidence_p':   psource[k]['incidence_p'],
            }
            out.append(ps)
        return out

    # ------------------------------------------------------------------
    # Generate source mesh
    # ------------------------------------------------------------------
    def _mesh(self, ps_centroid: list, ff_sd, ff_mo, ff_activation,
          ff_rupture_time, sett: dict,
          ff_rake=None, ff_dip=None) -> list:

        active   = [i for i in range(len(ff_mo)) if ff_mo[i] > 0.]
        N_active = len(active)
        sett     = sett.copy()
        sett['N_psources'] = N_active

        sources = []
        for i in active:
            s = sett.copy()
            s['Mo_ij']       = ff_mo[i]
            s['activation']  = ff_activation[i]
            s['stress_drop'] = ff_sd[i]
            s['ps_centroid'] = ps_centroid[i]
            s['t_rupture']   = ff_rupture_time[i]
            if ff_rake is not None:
                s['rake'] = ff_rake[i]
            if ff_dip is not None:
                s['dip']  = ff_dip[i]

            # ── Per-patch Afshari duration ────────────────────────────
            if self.dur_cfg.use_afshari:
                from .physics import afshari_stewart_2016

                # Sub-source equivalent magnitude from its moment
                Mw_ij = (2. / 3.) * np.log10(ff_mo[i]) - 10.7

                # Sub-source distance to site
                R_rup_ij = float(np.linalg.norm(
                    np.array(ps_centroid[i]) - np.array(s['site_coord'])
                ))

                # Rake for this patch (may be per-patch override)
                rake_ij = s['rake']

                D595_ij, _, _, _ = afshari_stewart_2016(
                    Mw   = Mw_ij,
                    Rrup = R_rup_ij,
                    rake = rake_ij,
                    vs30 = self.dur_cfg.vs30_afshari,
                    z1pt0= self.dur_cfg.z1pt0_afshari,
                    imt  = 'rsd595'
                )

                # Override duration for this patch only
                s['boolean_tgm_source'] = True
                s['Tgm_source']         = D595_ij
                s['boolean_tgm_path']   = True
                s['Tgm_path']           = 0.    # path already in D595
            # ──────────────────────────────────────────────────────────

            sources.append(s)

        n_cpu = self.sim_cfg.n_processors
        if n_cpu > 1:
            print(f"  [Subsources] Building {N_active} point sources "
                f"on {n_cpu} processes ...", flush=True)
            pool = mp.Pool(processes=n_cpu, maxtasksperchild=200)
            PSource = pool.map(_build_point_source, sources, chunksize=25)
            pool.close()
            pool.join()
            print(f"  [Subsources] Done — {N_active} sub-sources built.",
                flush=True)
        else:
            print(f"  [Subsources] Building {N_active} point sources "
                f"(sequential) ...", flush=True)
            PSource = []
            for k, s in enumerate(sources):
                PSource.append(PointSource(s).ps_info()[0])
                pct = (k + 1) / N_active * 100.
                if (k + 1) % max(1, N_active // 20) == 0 or (k + 1) == N_active:
                    print(f"    {k+1}/{N_active}  ({pct:.0f}%)", flush=True)
            print(f"  [Subsources] Done — {N_active} sub-sources built.",
                flush=True)

        return PSource


    @staticmethod
    def _rotate_plane(points: list, angle: float, axis: str) -> list:
        """Rotate a list of 3-D points around *axis* by *angle* [degrees]."""
        a = angle * math.pi / 180.
        ca, sa = math.cos(a), math.sin(a)
        if axis == 'x':
            R = np.array([[1., 0.,  0.],
                          [0., ca, -sa],
                          [0., sa,  ca]])
        elif axis == 'z':
            R = np.array([[ ca, -sa, 0.],
                          [ sa,  ca, 0.],
                          [ 0.,  0., 1.]])
        else:
            raise ValueError(f"Unknown rotation axis: '{axis}'")
        return [list(R @ np.array(p)) for p in points]
    
    # ------------------------------------------------------------------
    # Finite-fault geometry - Dictionary 
    # ------------------------------------------------------------------
    def set_geometry_from_dict(self, subsources):
        N = len(subsources)
        centroids = []
        for ss in subsources:
            coord = ss['centroid']
            if self.local_coords:
                centroids.append(np.array(coord, dtype=float))
            else:
                local = locate_site(
                    self.hypocenter_geographic,   # ← original [lon, lat, depth]
                    coord,                         # ← patch [lon, lat, depth_km]
                    reference=[0., 0.],            # ← local origin is [0,0]
                    depth=True
                )
                centroids.append(np.array(local, dtype=float))

        # 2. Moment — slip-weighted
        slips = np.array([ss['slip'] for ss in subsources], dtype=float)
        ff_mo = self.Mo * slips / slips.sum()

        # 3. Rupture times
        ff_rupture_time = np.array([ss['rupt_t'] for ss in subsources], dtype=float)

        # 4. Activation order — rank by rupture arrival time
        order = np.argsort(ff_rupture_time)
        ff_activation = np.empty(N, dtype=int)
        ff_activation[order] = np.arange(1, N + 1)

        # 5. Per-patch stress drop — fall back to global
        ff_sd = np.array([
            ss.get('sd', self.src_cfg.stress_drop) for ss in subsources
        ], dtype=float)

        # 6. Per-patch rake and dip — fall back to global
        ff_rake = [ss.get('rake', self.src_cfg.rake) for ss in subsources]
        ff_dip  = [ss.get('dip',  self.src_cfg.dip)  for ss in subsources]

        return centroids, ff_mo, ff_sd, ff_activation, ff_rupture_time, ff_rake, ff_dip

    # ------------------------------------------------------------------
    # Noise convolution
    # ------------------------------------------------------------------
    def add_noise(self, params: dict) -> list:
        PSource        = params['PSource']
        sim_seed       = params['sim_seed']
        dt             = params['dt']
        standard_freqs = params['standard_freqs']
        duration_max   = params['duration_max']
        stddev         = params['stddev']

        args = [
            (ps, sim_seed[i], dt, standard_freqs, duration_max, stddev)
            for i, ps in enumerate(PSource)
        ]

        n_cpu = self.sim_cfg.n_processors
        if n_cpu > 1:
            pool = mp.Pool(processes=n_cpu, maxtasksperchild=200)
            point = pool.map(_process_one_subsource, args, chunksize=50)
            pool.close()
            pool.join()

        else:
            point = []
            for i, a in enumerate(args):
                point.append(_process_one_subsource(a))
                pct = (i + 1) / len(args) * 100.
                if (i + 1) % max(1, len(args) // 20) == 0 or (i + 1) == len(args):
                    print(f"    [Convolution] {i+1}/{len(args)} ({pct:.0f}%)",
                        flush=True)
        return point

    # ------------------------------------------------------------------
    # Signal aggregation
    # ------------------------------------------------------------------
    def aggregate_signal(self, point: list):
        """
        Rotate sub-source signals from radial/transverse/vertical to
        North-South, East-West, Up-Down and sum coherently.
        """
        dt        = point[0]['dt']
        n_samples = point[0]['n_samples']
        N_ps      = len(point)

        NS = np.zeros(n_samples)
        EW = np.zeros(n_samples)
        UD = np.zeros(n_samples)

        for i in range(N_ps):
            phi = point[i]['phi']
            cp, sp = np.cos(phi), np.sin(phi)

            P_NS  =  point[i]['P_radial']  * cp
            P_EW  =  point[i]['P_radial']  * sp
            P_UD  =  point[i]['P_vertical']

            SV_NS =  point[i]['SV_radial'] * cp
            SV_EW =  point[i]['SV_radial'] * sp
            SV_UD =  point[i]['SV_vertical']

            SH_NS = -point[i]['SH'] * sp
            SH_EW =  point[i]['SH'] * cp

            def _trim(arr): return arr[:n_samples]

            if self.sim_cfg.sh_only:
                NS += _trim(SH_NS)
                EW += _trim(SH_EW)
            else:
                NS += _trim(P_NS) + _trim(SV_NS) + _trim(SH_NS)
                EW += _trim(P_EW) + _trim(SV_EW) + _trim(SH_EW)
                UD += _trim(P_UD) + _trim(SV_UD)

        return NS, EW, UD

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_duration_cap(win: np.ndarray, duration_max, n_samples: int,
                             dt: float) -> np.ndarray:
        if duration_max is not None:
            cap = int(duration_max / dt)
            if cap < n_samples:
                win = win[:cap]
        return win

    @staticmethod
    def _compute_time(n_samples: int, duration_max, dt: float) -> np.ndarray:
        if duration_max is not None:
            cap = int(duration_max / dt)
            if cap < n_samples:
                return np.linspace(0., cap * dt, cap)
        return np.linspace(0., n_samples * dt, n_samples)

# -*- coding: utf-8 -*-
"""
point_source.py
---------------
PointSource class: computes the frequency-domain ground-motion amplitude
spectra (FAS) for a single point source (sub-fault) and the associated
signal duration.

Supported source model
----------------------
Only the SCF (single-corner-frequency) model is supported.
Corner-frequency evolution is one of: 'Motazedian', 'RIK', 'Dang'.

Velocity units
--------------
All velocities throughout this module (vs, vp, alpha, betha) are in **km/s**.
Depths and distances (z, R_s, R_p, R_hyp) are in **km**.

Corner-frequency formula (Boore 2003 / Brune 1970, as used in SMSIM)
----------------------------------------------------------------------
    fc [Hz] = 0.4906 * beta [km/s] * (Δσ [bar] / M0 [dyne·cm])^(1/3)

Reference: Boore (2003, Pure Appl. Geophys.), SMSIM relation
    Δσ · M0 · fc  =  0.4906 · β³
with β in km/s, M0 in dyne·cm, Δσ in bars.

NOTE: a figure of "4.9×10⁶" sometimes appears in the literature; that form
uses β in cm/s (old CGS convention: β_cm/s = β_km/s × 1e5, so
0.4906 × 1e-5 × β_cm/s × (1e5)^1 gives the same result when stress drop
and moment are in dyne/cm² and dyne·cm respectively).  Since this code uses
β in km/s, M0 in dyne·cm, and Δσ in bars, **use 0.4906**.

Bug fixes vs. original Point_Source.py
---------------------------------------
* ``temp_vs`` NameError  : replaced with ``vs_medium``.
* ``build_spectra``      : ``wave_type is 'SV'`` / ``'SH'`` / ``'P'`` replaced
                           with ``==`` (identity comparison → equality).
* ``path()``             : Q factor branch now correctly checks for
                           ``wave_type in ('SV', 'SH')`` instead of ``== 'S'``
                           which was never True when called from build_spectra.
* ``mw_ij``              : kept as 2/3 (Python 3 float division is correct),
                           made explicit with ``2. / 3.``.
* Distance for path attenuation: controlled by PathConfig.use_ray_path.
  Default is hypocentral distance (same for P and S).  When use_ray_path=True,
  the ray-traced lengths R_p and R_s are used instead.
"""

from __future__ import annotations
import math as m
import copy
import numpy as np
import scipy.integrate
import scipy.interpolate

from .physics import (
    ray_propagation_bisection,
    FS_pwaves,
    FS_swaves,
    density_profile,
    vector_plane_angle,
    site_amplification,
    site_attenuation,
)


class PointSource:
    """
    Compute the amplitude spectra and travel times for one point source.

    Parameters
    ----------
    sett : dict
        Settings dictionary assembled by the EQ class.  All keys documented
        in ``EQ._build_sett``.

    After construction, call ``ps_info()`` to retrieve the output dictionary.
    """

    # ------------------------------------------------------------------
    def __init__(self, sett: dict):
        self._unpack(sett)
        self._ray_geometry()
        self._apply_manual_incidence()
        self.mw_ij = 2. / 3. * np.log10(self.Mo_ij) - 10.7
        self._compute_constants()
        self._compute_source_spectrum()
        spectra = self._build_all_spectra(sett['scaling_factor'],
                                          sett['TF_GIT'], sett['TF_GIT_s'], sett['TF_GIT_p'])
        duration = self._compute_duration(sett)
        self._assemble_output(spectra, duration, sett['t_rupture'])

    # ------------------------------------------------------------------
    # Input unpacking
    # ------------------------------------------------------------------
    def _unpack(self, sett: dict):
        # --- Source ---
        self.Mo_ij          = sett['Mo_ij']
        self.activation     = sett['activation']
        self.Mo             = sett['Mo']
        self.Moave          = sett['Moave']
        self.N_psources     = sett['N_psources']
        self.dip            = sett['dip']
        self.rake           = sett['rake']
        self.stress_drop_target = sett['stress_drop_target']
        self.stress_drop    = sett['stress_drop']
        self.stress_drop_p  = sett['stress_drop_p']
        self.stress_drop_s  = sett['stress_drop_s']
        self.gamma_p        = sett['gamma_p']
        self.gamma_s        = sett['gamma_s']
        self.F_pulse        = sett['F_pulse']
        self.fc_p           = sett['fc_p']
        self.fc_s           = sett['fc_s']
        self.evolutionary_frequency_model = sett['evolutionary_frequency_model']
        self.kappa_source   = sett['kappa_source']
        self.fcp_multiplier = sett['fcp_multiplier']

        # --- Medium ---
        # All velocities are in km/s; depths in km.
        self.depth_medium   = sett['depth']
        self.vs_medium      = sett['vs']       # km/s
        self.vp_medium      = sett['vp']       # km/s  (or None)
        self.rho_medium     = sett['rho']
        self.velocity_ratio = sett['velocity_ratio']

        # --- Path --
        self.R0, self.R1, self.b1 = sett['R0'], sett['R1'], sett['b1']
        self.R2, self.b2, self.b3 = sett['R2'], sett['b2'], sett['b3']
        self.use_ray_path   = sett['use_ray_path']
        self.Qo, self.Q1, self.Q1exp = sett['Qo'], sett['Q1'], sett['Q1exp']
        self.Qp_factor      = sett['Qp_factor']
        self.Att_non_par    = sett['Att_non_par']
        self.Att_non_par_p  = sett['Att_non_par_p']
        self.Att_non_par_s  = sett['Att_non_par_s']

        # --- Site ---
        self.amp_freq       = sett['amp_freq']
        self.amp            = sett['amp']
        self.kappa          = sett['kappa']
        self.kappa_p        = sett['kappa_p']
        self.kappa_s        = sett['kappa_s']
        self.fk             = sett['fk']
        self.fk_p           = sett['fk_p']
        self.fk_s           = sett['fk_s']
        self.f_max          = sett['f_max']
        self.f_max_exp      = sett['f_max_exp']
        self.site_lowpass   = sett['site_lowpass']

        # --- Simulation ---
        self.freq_ps        = sett['freq_ps']
        self.dt             = sett['dt']
        self.gm_type        = sett['gm_type']
        self.ps_centroid    = sett['ps_centroid']
        self.site_coord     = sett['site_coord']
        self.incidence_manual = sett['incidence_manual']

    # ------------------------------------------------------------------
    # Geometry and ray tracing
    # ------------------------------------------------------------------
    def _ray_geometry(self):
        """
        Compute hypocentral distance, ray-traced distances R_s / R_p,
        incidence angles, and travel times.

        All distances are in km; velocities in km/s.
        """
        ps_centroid = self.ps_centroid

        # --- Hypocentral distance [km] — default attenuation distance ---
        # Used for both geometrical spreading and anelastic attenuation unless
        # PathConfig.use_ray_path=True, in which case the ray-traced distances
        # R_s and R_p (computed below) are used instead.
        # R_s and R_p are always computed for reference (travel times, incidence
        # angles) regardless of which distance mode is active.
        self.R_hyp = np.linalg.norm(
            np.array(ps_centroid) - np.array(self.site_coord)
        )

        # --- Azimuth ---
        pos          = [ps_centroid[0] - self.site_coord[0],
                        ps_centroid[1] - self.site_coord[1]]
        self.azimuth = vector_plane_angle([0., 1.], pos)

        # --- Build layered medium up to source depth ---
        # Depths [km], velocities [km/s], density [g/cm³]
        depth_medium = np.asarray(self.depth_medium)
        vs           = np.asarray(self.vs_medium)    # km/s
        if self.vp_medium is None:
            vp = vs / self.velocity_ratio             # km/s
        else:
            vp = np.asarray(self.vp_medium)           # km/s
        rho = density_profile(vs) if self.rho_medium is None else np.asarray(self.rho_medium)

        if depth_medium[-1] < ps_centroid[2]:
            raise ValueError(
                f"Medium description (max depth {depth_medium[-1]} km) does not "
                f"reach the source depth ({ps_centroid[2]} km)."
            )

        idx_src     = np.where(depth_medium >= ps_centroid[2])[0][0]
        self.betha  = vs[idx_src - 1]   # S-wave velocity at source [km/s]
        self.alpha  = vp[idx_src - 1]   # P-wave velocity at source [km/s]
        self.rho_s  = rho[idx_src - 1]  # density at source [g/cm³]

        self.vs  = vs[:idx_src]          # [km/s]
        self.vp  = vp[:idx_src]          # [km/s]
        self.rho = rho[:idx_src]
        self.z   = np.append(depth_medium[:idx_src], ps_centroid[2])  # [km]

        # --- Ray tracing (results in km and km/s) ---
        data_s   = {'z': self.z, 'v': self.vs, 'start_point': ps_centroid,
                    'target': self.site_coord}
        result_s = ray_propagation_bisection(1e-30, m.pi / 2., 100, 0.01, data_s)
        if result_s is None:
            raise RuntimeError("S-wave ray propagation failed – no solution found.")
        self.R_s, self.theta_s, self.to_s = result_s   # R_s in km

        data_p   = {'z': self.z, 'v': self.vp, 'start_point': ps_centroid,
                    'target': self.site_coord}
        result_p = ray_propagation_bisection(1e-30, m.pi / 2., 100, 0.01, data_p)
        if result_p is None:
            raise RuntimeError("P-wave ray propagation failed – no solution found.")
        self.R_p, self.theta_p, self.to_p = result_p   # R_p in km

    def _apply_manual_incidence(self):
        if self.incidence_manual is not None:
            angle = self.incidence_manual
            angle = 1e-4 if angle == 0. else angle
            angle = 89.9 if angle == 90. else angle
            self.theta_s = angle * m.pi / 180.
            self.theta_p = angle * m.pi / 180.

    # ------------------------------------------------------------------
    # Spectral constants
    # ------------------------------------------------------------------
    def _compute_constants(self):
        """
        Radiation patterns, free-surface factors, energy-partition factors,
        and the scalar spectrum constants C.

        The constant C0 = Mo_ij / (4π ρ_s R0) × 1e-20.
        The 1e-20 arises from the mixed CGS/SI unit system:
          - Mo_ij   [dyne·cm]
          - rho_s   [g/cm³]
          - R0      [km]  → multiply by 1e5 to get cm  ⟹  factor 1e5
          - alpha, betha [km/s] → multiply by 1e5 to get cm/s ⟹  factor 1e15
          - combined: 1/(1e5 × 1e15) = 1e-20
        """
        dip_r  = self.dip  * m.pi / 180.
        rake_r = self.rake * m.pi / 180.

        RP_p  = m.sqrt(4. / 15.)
        RP_sv = 0.5 * m.sqrt(
            m.sin(rake_r)**2. * (14. / 15. + 1. / 3. * m.sin(2. * dip_r)**2.) +
            m.cos(rake_r)**2. * (4.  / 15. + 2. / 3. * m.cos(dip_r)**2.)
        )
        RP_sh = 0.55

        FS_p_v,  FS_p_r  = FS_pwaves(self.theta_p, self.alpha, self.betha)
        FS_sv_v, FS_sv_r = FS_swaves(self.theta_s, self.alpha, self.betha)
        FS_sh   = 2.

        EP_p_r  = -m.sin(self.theta_p)
        EP_p_v  =  m.cos(self.theta_p)
        EP_sv_r =  m.cos(self.theta_s)
        EP_sv_v =  m.sin(self.theta_s)
        EP_sh   = 1.

        # R0 [km], alpha/betha [km/s], rho_s [g/cm³] — unit factor 1e-20
        C0 = self.Mo_ij / (4. * m.pi * self.rho_s * self.R0) * 1e-20

        self.C_p_vertical  = RP_p  * FS_p_v  * EP_p_v  * C0 / self.alpha**3.
        self.C_p_radial    = RP_p  * FS_p_r  * EP_p_r  * C0 / self.alpha**3.
        self.C_sv_vertical = RP_sv * FS_sv_v * EP_sv_v * C0 / self.betha**3.
        self.C_sv_radial   = RP_sv * FS_sv_r * EP_sv_r * C0 / self.betha**3.
        self.C_sh          = RP_sh * FS_sh   * EP_sh   * C0 / self.betha**3.

        #print(f"DEBUG R0={self.R0}, R_hyp={self.R_hyp:.2f} km")
        #print(f"DEBUG C_sh={self.C_sh:.4e}, C_sv_radial={self.C_sv_radial:.4e}")
        #print(f"DEBUG rho_s={self.rho_s:.3f}, alpha={self.alpha:.3f}, betha={self.betha:.3f}")


    # ------------------------------------------------------------------
    # Source spectrum  (SCF model only)
    # ------------------------------------------------------------------
    def _compute_source_spectrum(self):
        """
        Compute per-sub-source corner frequencies, source spectra, and
        the incoherent-summation scaling factors H_s_ij, H_p_ij.

        Corner-frequency formula (Boore 2003 / Brune 1970, SMSIM convention):
            fc [Hz] = 0.4906 * beta [km/s] * (Δσ [bar] / M0 [dyne·cm])^(1/3)

        This follows directly from Δσ · M0 · fc = 0.4906 · β³ (Boore 2003).
        """
        f = self.freq_ps

        # Source-side attenuation (kappa at source)
        src_att = (np.ones_like(f) if self.kappa_source is None
                   else np.exp(-m.pi * self.kappa_source * f))

        # Event-level corner frequencies (may already be set by EQ._side_computations)
        if self.fc_s is None:
            sd_s = self.stress_drop_s if self.stress_drop_s is not None else self.stress_drop_target
            sd_p = self.stress_drop_p if self.stress_drop_p is not None else self.stress_drop_target
            # Boore (2003) SMSIM constant: 0.4906 with β in km/s, Δσ in bars, M0 in dyne·cm
            self.fc_s = 4.9e6* self.betha * (sd_s / self.Mo) ** (1. / 3.)
            self.fc_p = (
                self.alpha * self.fc_s / self.betha
                if self.fcp_multiplier is None
                else self.fcp_multiplier * self.fc_s
            ) if self.fc_p is None else self.fc_p

        # ----------------------------------------------------------------
        # Motazedian (2005) – evolutionary fc via pulse fraction
        # ----------------------------------------------------------------
        if self.evolutionary_frequency_model == 'Motazedian':
            if self.N_psources == 1:
                self.fc_s_ij = self.fc_s
                self.fc_p_ij = self.fc_p
            else:
                self.fc_s_ij = 4.9e6 * self.betha * (
                    self.stress_drop_target /
                    (self.Mo * min(self.activation / float(self.N_psources),
                                   self.F_pulse / 100.))
                ) ** (1. / 3.)
                self.fc_p_ij = (self.alpha * self.fc_s_ij / self.betha
                                if self.fcp_multiplier is None
                                else self.fcp_multiplier * self.fc_s_ij)

            self.source_ps_s = 1. / (1. + (f / self.fc_s_ij) ** self.gamma_s) * src_att
            self.source_ps_p = 1. / (1. + (f / self.fc_p_ij) ** self.gamma_p) * src_att
            self.H_s_ij      = self.N_psources ** 0.5 * (self.fc_s / self.fc_s_ij) ** 2.
            self.H_p_ij      = self.N_psources ** 0.5 * (self.fc_p / self.fc_p_ij) ** 2.

        # ----------------------------------------------------------------
        # RIK – fc from local sub-source stress drop
        # ----------------------------------------------------------------
        elif self.evolutionary_frequency_model == 'RIK':
            if self.N_psources == 1:
                self.fc_s_ij = self.fc_s
                self.fc_p_ij = self.fc_p
            else:
                self.fc_s_ij = 4.9e6 * self.betha * (
                    self.stress_drop / self.Mo_ij
                ) ** (1. / 3.)
                self.fc_p_ij = (self.alpha * self.fc_s_ij / self.betha
                                if self.fcp_multiplier is None
                                else self.fcp_multiplier * self.fc_s_ij)

            self.source_ps_s = 1. / (1. + (f / self.fc_s_ij) ** 2.) * src_att
            self.source_ps_p = 1. / (1. + (f / self.fc_p_ij) ** 2.) * src_att
            self.H_s_ij      = self.N_psources ** 0.5 * (self.fc_s / self.fc_s_ij) ** 2.
            self.H_p_ij      = self.N_psources ** 0.5 * (self.fc_p / self.fc_p_ij) ** 2.

        # ----------------------------------------------------------------
        # Dang (2021) – fc scaled by sub-source moment ratio
        # ----------------------------------------------------------------
        elif self.evolutionary_frequency_model == 'Dang':
            if self.N_psources == 1:
                self.fc_s_ij = self.fc_s
                self.fc_p_ij = self.fc_p
            else:
                self.fc_s_ij = self.fc_s * (1. + (self.Mo_ij / self.Moave) ** (1. / 3.))
                self.fc_p_ij = (self.alpha * self.fc_s_ij / self.betha
                                if self.fcp_multiplier is None
                                else self.fcp_multiplier * self.fc_s_ij)

            self.source_ps_s = 1. / (1. + (f / self.fc_s_ij) ** 2.) * src_att
            self.source_ps_p = 1. / (1. + (f / self.fc_p_ij) ** 2.) * src_att

            # Energy-preserving scaling (Dang et al., 2021)
            self.H_s_ij = self._energy_scaling(f, self.fc_s,  self.fc_s_ij)
            self.H_p_ij = self._energy_scaling(f, self.fc_p,  self.fc_p_ij)

        else:
            raise ValueError(
                f"Unknown evolutionary_frequency_model: "
                f"'{self.evolutionary_frequency_model}'. "
                f"Must be one of 'Motazedian', 'RIK', 'Dang'."
            )

    def _energy_scaling(self, f, fc_global: float, fc_local: float) -> float:
        """Return the Dang-style energy-preserving scaling factor H."""
        f_static = (f / (1. + (f / fc_global) ** 2.)) ** 2.
        f_ij     = (f / (1. + (f / fc_local)  ** 2.)) ** 2.
        I_static = scipy.integrate.simpson(f_static, x=f)
        I_ij     = scipy.integrate.simpson(f_ij,     x=f)
        return m.sqrt(I_static / I_ij * self.Mo / self.Mo_ij)

    # ------------------------------------------------------------------
    # Build spectra
    # ------------------------------------------------------------------
    def _build_all_spectra(self, scaling_factor, TF_GIT, TF_GIT_s, TF_GIT_p):
        """
        Build the five component spectra and apply GIT transfer functions.

        NOTE on GIT transfer functions
        --------------------------------
        The TF applied here (via _apply_git_tf) is an **amplitude-only**
        transfer function derived from GIT inversion.  It modifies the
        deterministic FAS shape before noise convolution.
        The TF parameter inside scale_spectrum (in EQ.add_noise) is a
        **complex transfer function** (amplitude + phase), intended for
        cases where the full propagation response — including phase delays —
        is known empirically.  The two mechanisms are complementary and
        should not be confused.
        """
        A = {}
        A['p_vertical']  = self._build_spectra(self.C_p_vertical,  self.source_ps_p, 1., 'P',  self.Att_non_par_p)
        A['p_radial']    = self._build_spectra(self.C_p_radial,    self.source_ps_p, 1., 'P',  self.Att_non_par_p)
        A['sv_vertical'] = self._build_spectra(self.C_sv_vertical, self.source_ps_s, 1., 'SV', self.Att_non_par_s)
        A['sv_radial']   = self._build_spectra(self.C_sv_radial,   self.source_ps_s, 1., 'SV', self.Att_non_par_s)
        A['sh']          = self._build_spectra(self.C_sh,          self.source_ps_s, 1., 'SH', self.Att_non_par_s)

        for k in A:
            A[k] = A[k] * scaling_factor

        # Amplitude-only GIT transfer functions
        tf_s = TF_GIT_s if TF_GIT_s is not None else TF_GIT
        if tf_s is not None:
            A['sh']        = self._apply_git_tf(A['sh'],        tf_s)
            A['sv_radial'] = self._apply_git_tf(A['sv_radial'], tf_s)
        if TF_GIT_p is not None:
            A['p_vertical'] = self._apply_git_tf(A['p_vertical'], TF_GIT_p)

        return A

    def _build_spectra(self, C: float, source_spectrum: np.ndarray,
                       H: float, wave_type: str,
                       Att_non_par=None) -> np.ndarray:
        """
        Assemble the amplitude spectrum for one polarisation.

        Parameters
        ----------
        C              : scalar spectrum constant (includes Mo, RP, FS, EP, rho, R0, velocity)
        source_spectrum: normalised source shape S(f)
        H              : incoherent-summation scaling factor H_ij
        wave_type      : 'P', 'SV', or 'SH'
        Att_non_par    : non-parametric attenuation table (or None)
        """
        f = self.freq_ps

        # Ground-motion type integrator: (2πf)^I_exp
        I_exp = 2. if self.gm_type == 'acc' else 1. if self.gm_type == 'vel' else 0.
        I     = (2. * m.pi * f) ** I_exp

        # Path attenuation distance:
        #   use_ray_path=False (default) → hypocentral distance, identical
        #                                  for P and S waves
        #   use_ray_path=True            → true ray path length, differs
        #                                  between P and S because the waves
        #                                  travel at different velocities
        if self.use_ray_path:
            R = self.R_p if wave_type == 'P' else self.R_s
        else:
            R = self.R_hyp   # same for P and S

        P = self._path(wave_type, R, Att_non_par=Att_non_par)

        # Site kappa — wave-type-specific overrides the broadband kappa
        kappa = (self.kappa_p if (wave_type == 'P'  and self.kappa_p is not None) else
                 self.kappa_s if (wave_type != 'P'  and self.kappa_s is not None) else
                 self.kappa)

        # Site fk — wave-type-specific pair takes priority; then 'corner' alias
        if self.fk_s is not None and self.fk_p is not None:
            fk = self.fk_p if wave_type == 'P' else self.fk_s
        elif self.fk == 'corner':
            fk = self.fc_p_ij if wave_type == 'P' else self.fc_s_ij
        else:
            fk = self.fk

        G = self._site_filter(kappa, fk)

        # Incoherent-summation correction (Boore, 2009)
        c       = self.N_psources ** 0.5 / H
        fo_sf   = self.fc_p_ij if wave_type == 'P' else self.fc_s_ij
        fo_eff  = fo_sf / c ** 0.5
        s       = c * (1. + (f / fo_sf) ** 2.) / (1. + (f / fo_eff) ** 2.)
        source_spectrum = source_spectrum * s

        #print(f"DEBUG wave={wave_type},  max(A)={float(np.max(np.abs(C*P*G*I*source_spectrum*H))):.4e}")

        return C * P * G * I * source_spectrum * H

    def _site_filter(self, kappa, fk) -> np.ndarray:
        """G(f) = Amp(f) * D(f)."""
        Amp = site_amplification(self.freq_ps, self.amp_freq, self.amp)
        D   = site_attenuation(self.freq_ps, kappa=kappa, f_max=self.f_max,
                                lowpass=self.site_lowpass, f_max_exp=self.f_max_exp, fk=fk)
        return Amp * D

    def _apply_git_tf(self, A: np.ndarray, tf: dict) -> np.ndarray:
        """
        Multiply a deterministic amplitude spectrum by an empirical GIT
        amplitude-only transfer function.

        The TF dict must contain keys 'freqs' and 'tf'.
        """
        fn = scipy.interpolate.interp1d(
            tf['freqs'], tf['tf'], kind='linear',
            bounds_error=False, fill_value=(tf['tf'][0], tf['tf'][-1])
        )
        return A * fn(self.freq_ps)

    # ------------------------------------------------------------------
    # Path attenuation
    # ------------------------------------------------------------------
    def _path(self, wave_type: str, distance: float,
              Att_non_par=None) -> np.ndarray:
        """
        Piecewise geometrical spreading + anelastic attenuation.

        Distance and R0, R1, R2 must be in consistent units [km].
        Velocities alpha/betha are in km/s.

        Parameters
        ----------
        wave_type   : 'P', 'SV', or 'SH'
        distance    : propagation distance [km]
        Att_non_par : non-parametric attenuation table or None
        """
        f = self.freq_ps

        if Att_non_par is None:
            # --- Geometrical spreading ---
            R0, R1, R2 = self.R0, self.R1, self.R2
            b1, b2, b3 = self.b1, self.b2, self.b3
            if distance <= R1:
                Z = (R0 / distance) ** b1
            elif distance <= R2:
                Z = (R0 / R1) ** b1 * (R1 / distance) ** b2
            else:
                Z = (R0 / R1) ** b1 * (R1 / R2) ** b2 * (R2 / distance) ** b3

            # --- Anelastic attenuation ---
            is_s_wave = wave_type in ('SV', 'SH')
            if is_s_wave:
                q_factor = 1.0
            else:
                q_factor = ((3. / 4.) * (self.alpha / self.betha) ** 2.
                            if self.Qp_factor == 'udias'
                            else float(self.Qp_factor))

            # Q(f) = max(Qo, Q1 * f^Q1exp) [dimensionless]
            # c [km/s] × Q [–] → the exponent π f R / (Q c) is dimensionless
            # when f [Hz], R [km], c [km/s].
            Q = np.maximum(self.Qo, self.Q1 * f ** self.Q1exp) * q_factor
            c = self.betha if is_s_wave else self.alpha   # km/s
            P = Z * np.exp(-m.pi * f * (distance - self.R0) / (Q * c))

        else:
            # --- Non-parametric (GIT) attenuation ---
            Att, freqs_att, r_att = (Att_non_par['Att'],
                                     Att_non_par['freqs'],
                                     Att_non_par['R'])
            att = np.array([
                10. ** self._interpolate_attenuation(Att, freqs_att, r_att, fi, distance)
                for fi in f
            ])
            is_s_wave = wave_type in ('SV', 'SH')
            if isinstance(self.Qp_factor, float):
                q_factor = 1. if is_s_wave else self.Qp_factor
            else:
                q_factor = (1. if is_s_wave
                            else (3. / 4.) * (self.alpha / self.betha) ** 2.)
            P = att * q_factor

        return P

    def _interpolate_attenuation(self, Att, freqs_att, r_att,
                                  freq_int: float, r_int: float) -> float:
        """Bilinear interpolation of the non-parametric attenuation table."""
        freqs_att = np.asarray(freqs_att)
        if freq_int < freqs_att[0]:
            fi_lo, fi_hi = 0, 0
        elif freq_int > freqs_att[-1]:
            fi_lo, fi_hi = -1, -1
        else:
            fi_lo = int(np.searchsorted(freqs_att, freq_int, side='right') - 1)
            fi_hi = min(fi_lo + 1, len(freqs_att) - 1)

        total_dist = 0.
        points, dists = [], []
        for fi in [fi_lo, fi_hi]:
            R_freq = np.asarray(r_att[fi])
            if r_int <= R_freq[0]:
                ri_lo, ri_hi = 0, 0
            elif r_int >= R_freq[-1]:
                ri_lo, ri_hi = -1, -1
            else:
                ri_lo = int(np.searchsorted(R_freq, r_int, side='right') - 1)
                ri_hi = min(ri_lo + 1, len(R_freq) - 1)

            P1 = Att[fi][ri_lo]
            P2 = Att[fi][ri_hi]
            d1 = m.sqrt((freq_int - freqs_att[fi]) ** 2. + (r_int - R_freq[ri_lo]) ** 2.)
            d2 = m.sqrt((freq_int - freqs_att[fi]) ** 2. + (r_int - R_freq[ri_hi]) ** 2.)
            total_dist += d1 + d2
            points.append([P1, P2])
            dists.append([d1, d2])

        return (points[0][0] * dists[0][0] + points[0][1] * dists[0][1] +
                points[1][0] * dists[1][0] + points[1][1] * dists[1][1]) / total_dist

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------
    def _compute_duration(self, sett: dict):
        """
        Return (tgm_s, tgm_p) in seconds.

        Path duration uses the same distance selection as path attenuation:
        hypocentral (R_hyp) by default, or ray-traced (R_s, R_p) when
        use_ray_path=True.
        """
        boolean_tgm_source   = sett['boolean_tgm_source']
        Tgm_source           = sett['Tgm_source']
        boolean_tgm_path     = sett['boolean_tgm_path']
        Tgm_path             = sett['Tgm_path']
        path_duration        = sett['path_duration']
        ps_specific_duration = sett['ps_specific_duration']
        duration_swave       = sett['duration_swave']
        duration_pwave       = sett['duration_pwave']

        if ps_specific_duration:
            return duration_swave, duration_pwave

        # Source duration
        if boolean_tgm_source:
            t_source_s = t_source_p = Tgm_source
        else:
            if self.evolutionary_frequency_model == 'Motazedian':
                t_source_s = 1. / self.fc_s_ij
                t_source_p = 1. / self.fc_p_ij
            else:
                t_source_s = 10. ** (0.5 * self.mw_ij - 2.251)
                t_source_p = t_source_s

        # Path duration
        if boolean_tgm_path:
            t_path_s = t_path_p = Tgm_path
        else:
            if self.use_ray_path:
                t_path_s = path_duration * self.R_s
                t_path_p = path_duration * self.R_p
            else:
                t_path_s = path_duration * self.R_hyp
                t_path_p = path_duration * self.R_hyp

        return t_source_s + t_path_s, t_source_p + t_path_p

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _assemble_output(self, spectra: dict, duration, t_rupture: float):
        tgm_s, tgm_p = duration
        self.output = {
            'Mo_ij':         self.Mo_ij,
            'Mw_ij':         self.mw_ij,
            'A_p_vertical':  spectra['p_vertical'],
            'A_p_radial':    spectra['p_radial'],
            'A_sv_vertical': spectra['sv_vertical'],
            'A_sv_radial':   spectra['sv_radial'],
            'A_sh':          spectra['sh'],
            'frequencies':   self.freq_ps,
            'to_p':          self.to_p,
            'to_s':          self.to_s,
            't_rupture':     t_rupture,
            'tgm_p':         tgm_p,
            'tgm_s':         tgm_s,
            'fc_p_ij':       self.fc_p_ij,
            'fc_s_ij':       self.fc_s_ij,
            'H_p_ij':        self.H_p_ij,
            'H_s_ij':        self.H_s_ij,
            'azimuth':       self.azimuth,
            'incidence_p':   self.theta_p,
            'incidence_s':   self.theta_s,
        }

    def ps_info(self) -> list:
        """Return the output dictionary as a single-element list."""
        return [self.output]

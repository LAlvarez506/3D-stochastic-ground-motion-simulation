# Verification suite

Each script is a module run from the repository root (`python -m stochastic_gm.verification.<name>`),
prints a table, writes figures to `--output-dir` (default `./verification_output`) and exits with
status 1 if a pass/fail criterion is not met. Dependencies: numpy, scipy, matplotlib; the GMPE test
also needs `pip install pygmm`.

| # | Script | What it tests | Runtime (default) |
|---|--------|---------------|-------------------|
| 0 | `point_source_validation` | End-to-end point-source run: spectra, time series, response spectra | seconds |
| 1 | `verify_interfrequency_correlation` | The correlated noise reproduces the BA19 correlation structure (like paper Fig. 5) | minutes (n-sim 2000) |
| 2 | `verify_noise_scaled_spectra` | Deterministic spectrum vs noise-scaled spectra; amplitude scaling | minutes (n-sim 1000) |
| 3 | `verify_rvt` | Energy (Parseval) and peak motion by random vibration theory (Boore 2003) | ~10 s |
| 4 | `verify_gmpe_ab06` | PGA/PSA against the Atkinson & Boore (2006) GMPE, plus an independent Boore point-source/RVT model | ~30 s |

`rvt.py` holds the RVT helpers (spectral moments, Cartwright & Longuet-Higgins peak factor, Eq. 38
duration, SDOF response); `rvt.self_test()` checks them against Monte-Carlo stationary noise (~1 %).

## 1. Interfrequency correlation
```
python -m stochastic_gm.verification.verify_interfrequency_correlation --n-sim 2000
```
Simulates an ensemble, computes the smoothed EAS (Konno-Ohmachi, b = 188.5), estimates the correlation of
the normalized residuals and compares it with the model (`--correlation-type`, `--normalization`,
`--apply-to`, `--mw`). Outputs cross-sections at 0.2/0.5/2/5/15 Hz with 95 % bands, contour maps and a summary.
Reference: within-site RMSE 0.029 against ~0.022 of sampling noise.

## 2. Spectra scaled by the noise
```
python -m stochastic_gm.verification.verify_noise_scaled_spectra --n-sim 1000
```
Target EAS of the sub-source vs baseline and correlated records, per normalization (`expected`, `none`,
`per_realization`). Prints baseline rms/target (should be ~1; warning if off by >10 %).

## 3. Energy and RVT (`verify_rvt`)
```
python -m stochastic_gm.verification.verify_rvt --n-sim 100 [--ftgm 2.0]
```
* **Energy:** mean simulated energy of the two horizontals against m0 = 2 ∫|Y|² df of the deterministic
  horizontal spectrum (Boore 2003, Eqs. 35-36). Expected 1.00. Tolerance 5 %.
* **RVT peaks:** PGA and PSA from the same spectrum with RVT (Eqs. 29-38, Tgm of the simulator) against the
  mean simulated peak. Pass/fail (±25 %) on PGA and PSA 0.1/0.3 s; 1 s and 3 s are informational (RVT itself is
  only good to ~10-20 % there, also for stationary noise).
* `--ftgm` is Tw/Tgm of the window. Boore uses 2.0 (default of this test); the example scripts use 0.5, which packs
  the energy into about a quarter of the duration, so RVT with Tgm then overpredicts long-period PSA.

Result (Tw = 2 Tgm): PGA RVT/simulation 1.03-1.18; PSA 0.1 s 0.98-1.27; PSA 0.3 s 1.03-1.26.

**Known finding.** For large magnitude, short distance and long windows the energy ratio falls to 0.67-0.9 and the
script reports FAIL. Cause: P and S are built from the *same* white-noise sequence; the horizontal radial motion adds
P (-sin θp) and SV (+cos θs) with opposite signs, so the common part cancels when the P and S windows overlap. With
independent noise for P and S the ratio is 0.99-1.00 in every scenario (tested on a scratch copy, not yet applied
to the package).

## 4. GMPE comparison (`verify_gmpe_ab06`)
```
pip install pygmm
python -m stochastic_gm.verification.verify_gmpe_ab06 --n-sim 60
```
Parameters (Atkinson & Boore 2006; Atkinson 2004 attenuation model), ENA hard rock: Δσ = 140 bar, β = 3.7 km/s,
ρ = 2.8 g/cm³, Vp = √3 β, Q = max(1000, 893 f^0.32), κ = 0.005 s, hard-rock amplification (1.00 at 0.5 Hz,
1.13, 1.22, 1.36, 1.41 at 10 Hz), duration 1/fc + cumulative slopes 0 / 0.16 / -0.03 / 0.04 s/km (hinges 10, 70, 130 km),
geometric spreading hinged at 70 and 140 km (exponents -0.2 and 0.5 beyond), dt = 0.004 s, Mw 5 and 6, 15-150 km
hypocentral distance, GMPE = `pygmm.AtkinsonBoore2006` at Vs30 = 2000 m/s (geometric-mean horizontal, 5 % damping).

Three curves per panel: AB06, the simulator (ensemble median and ±1σ band) and an independent *textbook* point-source model
(radiation 0.55, free surface 2, partition 0.71) with RVT, which separates "simulator vs Boore model" from
"model vs GMPE".

Two variants of the near-source spreading:

| Variant | b1 (≤70 km) | Purpose |
|---------|-------------|---------|
| `ab06-literal` | 1.3 | Published exponent, referenced to 1 km. Informational. |
| `r-1-spreading` | 1.0 | R^-1 to 70 km as in Atkinson & Boore (1995) / Frankel et al. (1996). Gated. |

Criteria: (a) code check, simulator/independent-RVT within a factor 1.25 at PGA, PSA 0.1 and 0.2 s (median over distance);
(b) for `r-1-spreading`, simulator/AB06 within 1.35 at PGA and PSA 0.1 s.

Results (median over distance):

* Simulator / textbook RVT: PGA 1.08-1.09, PSA 0.1 s 0.91-0.93, 0.2 s 0.83-0.88 (lower at 0.5-1 s, where RVT is known to
  overestimate).
* `r-1-spreading`, simulator / AB06: PGA 1.07 (Mw 5), 1.23 (Mw 6); PSA 0.1 s 0.95, 1.11; PSA 0.2 s 1.06, 1.06; PSA 0.5 s 1.22, 1.12;
  PSA 1 s 1.27, 1.32. Within the GMPE's own ±1σ everywhere.
* `ab06-literal`: both simulator and independent RVT sit at 0.3-0.4 of AB06 for PGA and ~0.7 at 1 s, with a distance trend of
  roughly R^-0.3. Because two independent implementations agree, this is a parameter-equivalence issue (the AB06 GMPE is a
  regression to finite-fault simulations; an R^-1.3 spreading referenced to 1 km does not reproduce its level at short periods),
  not a simulator issue.

Caveats: AB06 is finite-fault derived, so a point source is only comparable at small and moderate magnitudes and not
very close to the fault; the simulator uses the physical radiation pattern, incidence angle and free-surface factors,
AB06 uses the average values.

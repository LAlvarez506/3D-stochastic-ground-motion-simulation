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
| 5 | `verify_finite_fault_postprocess` | Correlated noise imposed on summed finite-fault records (`postprocess`): correlation, sigma, energy, components | ~12 min (n-sim 800) |

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

Result (Tw = 2 Tgm, independent P/S noise, the default): energy ratio 0.97-1.01 in all nine scenarios; RVT/simulation for PGA
0.94-1.07, PSA 0.1 s 0.95-1.01, PSA 0.3 s 0.97-1.23; PASS.

**P/S noise option.** `SimConfig(independent_ps_noise=True)` (default) draws independent white-noise sequences for the P and S
waves. With `False` (original behaviour, `--shared-ps-noise` in this script) one sequence is shared; when the P and S windows overlap
(large magnitude, short distance, long window) the radial P (-sin θp) and SV (+cos θs) terms have opposite signs and partly cancel,
and the energy falls to 0.67-0.9 of m0 (the script then reports FAIL). The shared option is kept for sensitivity /
uncertainty-propagation studies.

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

* Simulator / textbook RVT: PGA 1.07-1.08, PSA 0.1 s 0.90-0.92, 0.2 s 0.83-0.88 (lower at 0.5-1 s, where RVT is known to
  overestimate).
* `r-1-spreading`, simulator / AB06 (Mw 5, Mw 6): PGA 1.06, 1.22; PSA 0.1 s 0.95, 1.10; PSA 0.2 s 1.07, 1.08; PSA 0.5 s 1.27, 1.19;
  PSA 1 s 1.36, 1.36. Within the GMPE's own ±1σ everywhere.
* `ab06-literal`: both simulator and independent RVT sit at 0.3-0.4 of AB06 for PGA and ~0.7 at 1 s, with a distance trend of
  roughly R^-0.3. Because two independent implementations agree, this is a parameter-equivalence issue (the AB06 GMPE is a
  regression to finite-fault simulations; an R^-1.3 spreading referenced to 1 km does not reproduce its level at short periods),
  not a simulator issue.

Caveats: AB06 is finite-fault derived, so a point source is only comparable at small and moderate magnitudes and not
very close to the fault; the simulator uses the physical radiation pattern, incidence angle and free-surface factors,
AB06 uses the average values.

## 5. Finite-fault post-processing (`verify_finite_fault_postprocess`)
```
python -m stochastic_gm.verification.verify_finite_fault_postprocess --n-sim 800 [--mode replace|scale] [--smoothing-b 10]
```
Scenario: synthetic planar fault (Mw 6.5, 20 x 10 km, 60° dip, 8 x 4 sub-sources, rupture speed 2.8 km/s), site 25 km
from the fault, homogeneous half space. Three ensembles of summed records are compared against the BA19 within-site model
(the same smoothed-EAS estimator as test 1, b = 188.5):

* **baseline** – independent noise per sub-source (no correlation);
* **per-subsource** – `SimConfig.corr_noise` on the finite fault (a correlated epsilon per sub-source, then summed);
* **post-processed** – baseline + `postprocess_eq(eq, CorrelatedNoiseConfig(...))`: one epsilon on the summed record.

Checks: (1) correlation RMSE <= 2/sqrt(n_sim); (2) mean ratio of the std of ln EAS to sigma(f; Mw) within 10 %;
(3) mean energy ratio post/original within 3 standard errors of 1 (normalization `expected`; the ratio is heavy-tailed
because the residual is strongly correlated in frequency); (4) vertical identical and NS/EW ratio unchanged.

Result (n_sim = 800): correlation RMSE baseline 0.383, per-subsource 0.355, **post-processed 0.052** (sampling noise 0.035);
std of ln EAS / target: per-subsource 0.86, **post-processed 1.06** (0.21-0.37 at 5-20 Hz for the others against targets 0.37-0.45);
mean energy ratio 0.956 (standard error 0.018); NS/EW energy ratio factor 0.996; vertical identical. PASS.

The per-subsource result shows why post-processing is needed: summing independent correlated draws removes the correlation
(the sum is dominated by the diagonal) and shrinks sigma at high frequency.

Notes on the method and its knobs:

* `smoothing_b` (default 10) sets the bandwidth of the reference spectrum. Wider smoothing (b = 5) recovers sigma and correlation a
  little better (RMSE 0.064 vs 0.072 at n_sim = 300, sigma ratio 1.00 vs 1.04), but flattens real spectral structure
  (directivity, sub-source corners); b = 20 gives RMSE 0.086 and sigma ratio 1.09. The smoothed spectrum is rescaled so the energy
  of each component is unchanged.
* `mode='scale'` (multiply the record's own spectrum by F) keeps the record's scatter, so the total variability is its own plus
  BA19: sigma ratio 1.3 and RMSE 0.16 here. Use it only if that is what you want to study.
* With `normalization='per_realization'` the energy of every record is exactly preserved (at the cost of distorting sigma and
  the correlation, as in the point source); `expected` preserves it on average; `none` leaves the median unchanged.
* Residual differences from the model are largest at 0.2-0.5 Hz, where the record has few frequency bins inside the reference
  window (about 23 s records).
* One epsilon is drawn per record, i.e. per site and per realization. Several sites of the same event are not correlated with each
  other by this tool (spatial correlation and the between-event component are out of scope).

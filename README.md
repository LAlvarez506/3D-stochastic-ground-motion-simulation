# stochastic_gm

Stochastic ground-motion simulator based on the Boore (2003) stochastic
method, extended to multi-component **(P, SV, SH)** output for point-source
and finite-fault models by Otarola et al (2016, 2018) and latter modified by Alvarez 2022.

---

## Package layout

```
stochastic_gm/                  ← repository root
│
├── setup.py
├── requirements.txt
├── README.md
├── examples/                   ← usage examples
│   ├── run_simulation_exemple.py        ← point-source simulation
│   ├── run_finite_fault.py              ← finite-fault simulation (dict-based)
│   └── lebanon_faille.csv               ← example fault model used by run_finite_fault.py
│
└── stochastic_gm/              ← importable package
    ├── __init__.py             ← public API (EQ, configs, physics helpers)
    ├── config.py               ← typed dataclass configurations
    ├── earthquake.py           ← EQ orchestrator class
    ├── point_source.py         ← PointSource spectral engine
    ├── physics.py              ← stateless numerical functions
    └── verification/
        ├── __init__.py
        └── point_source_validation.py   ← point-source verification suite
```

---

## Installation

### Requirements

- Python >= 3.8
- numpy >= 1.21
- scipy >= 1.7
- numba >= 0.55
- matplotlib >= 3.4 *(only needed for the example scripts)*
- pandas *(only needed for the example scripts)*

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Install the package

Clone the repository and install in editable mode (recommended during development — any edit to the source is immediately reflected without re-installing):

```bash
git clone https://github.com/YOUR_USERNAME/stochastic_gm.git
cd stochastic_gm
pip install -e .
```

Or install without cloning:

```bash
pip install .
```

---

## Getting started

Two ready-to-run example scripts are included at the root of the repository:

| Script | Description |
|---|---|
| `examples/run_simulation_exemple.py` | Point-source simulation — the simplest starting point |
| `examples/run_finite_fault.py` | Finite-fault simulation — loads `lebanon_faille.csv` and runs a full finite-fault simulation |

Run either script directly after installing the package:

```bash
python examples/run_simulation_exemple.py   # point-source
python examples/run_finite_fault.py         # finite-fault
```

Both scripts are fully annotated and cover the complete workflow: configuring the medium, source, path, site, and simulation parameters, running `EQ`, and plotting the output waveforms. The finite-fault script reads `examples/lebanon_faille.csv`, a semicolon-separated file with columns `lon`, `lat`, `depth` (m), `slip` (m), `rupture` (s), `dip` (deg), and `rake` (deg), one row per fault patch.

A validation suite for the point-source engine is also available as a module:

```bash
python -m stochastic_gm.examples.point_source_validation --Vp 6.0 --Vs 3.46 --depth 15 --dist 30 --Mw 6.0 --sd 50 --show
```

---

## Finite-fault mode

In finite-fault mode the rupture is discretised into a list of sub-sources, each described by a plain Python dict. The tool handles moment partitioning, activation ordering, and coordinate conversion internally — you only need to supply the geometry and kinematics.

### Minimal example

```python
import numpy as np
from stochastic_gm import (
    EQ,
    MediumConfig, SourceConfig, PathConfig,
    SiteConfig, DurationConfig, SimConfig,
)

medium   = MediumConfig(
    system = 'local',
    Depth  = np.array([0., 30.]),
    vs     = np.array([3.46, 3.46]),
    vp     = np.array([6.00, 6.00]),
    Rho    = np.array([2.7,  2.7]),
)
source   = SourceConfig(stress_drop=50., rake=90., dip=45.)
path     = PathConfig(Qo=300., Q1=0., Q1exp=0., R0=1., R1=100., b1=1., R2=200., b2=0.5, b3=0.)
site     = SiteConfig(amp_freq=np.array([0.01, 200.]), amp=np.array([1.0, 1.0]))
duration = DurationConfig(path_duration=0.05)
sim      = SimConfig(dt=0.01, n_sim=1, seed=42,
                     epsilon_s=0.2, nu_s=0.05, ftgm_s=0.5,
                     epsilon_p=0.2, nu_p=0.05, ftgm_p=0.5)

# Three sub-sources along a simple fault — local Cartesian coordinates [x, y, depth] in km
subsource_list = [
    {'centroid': [ 0., 0., 12.], 'slip': 1.5, 'rupt_t': 0.0},
    {'centroid': [ 5., 0., 13.], 'slip': 2.8, 'rupt_t': 0.9},
    {'centroid': [10., 0., 14.], 'slip': 1.1, 'rupt_t': 1.8},
]

eq = EQ(
    Mw             = 6.5,
    medium         = medium,
    source         = source,
    path           = path,
    site_cfg       = site,
    duration       = duration,
    sim            = sim,
    hypocenter     = [0., 0., 12.],   # km — local coordinates
    site_coord     = [30., 0.,  0.],  # km
    point_source   = False,
    subsource_list = subsource_list,
)

NS, EW, UD = eq.sim_acc[0]
print(f"PGA_NS = {max(abs(NS)):.3f} cm/s²")
```

### Enabling finite-fault mode

Set `point_source=False` and pass `subsource_list` to `EQ`:

```python
eq = EQ(
    Mw             = 7.0,
    medium         = medium,
    source         = source,
    path           = path,
    site_cfg       = site_cfg,
    duration       = duration,
    sim            = sim,
    hypocenter     = hypocenter,
    site_coord     = site_coord,
    point_source   = False,        # ← switch to finite-fault
    subsource_list = subsource_list,
)
```

### Sub-source dict format

Each dict in `subsource_list` represents one fault patch:

```python
{
    'centroid': [lon, lat, depth_km],  # patch centre — geographic [lon, lat, depth_km]
                                       # or local Cartesian [x_km, y_km, z_km]
                                       # (controlled by MediumConfig.system)
    'slip'    :  slip_m,               # slip [m] — used to weight seismic moment
    'rupt_t'  :  rupture_time_s,       # rupture arrival time at this patch [s]

    # Optional — override the global values set in SourceConfig:
    'dip'     :  dip_deg,              # per-patch dip  [deg]
    'rake'    :  rake_deg,             # per-patch rake [deg]
    'sd'      :  stress_drop_bar,      # per-patch stress drop [bar]
}
```

Patches with zero slip are ignored internally. Seismic moment is distributed proportionally to slip, so only relative slip values matter. Activation order is derived automatically from `rupt_t`.

### Coordinate system

The coordinate system for `centroid`, `hypocenter`, and `site_coord` is set once via `MediumConfig.system`:

- `'geographic'` — pass `[longitude, latitude, depth_km]`; the tool converts to local Cartesian internally using the hypocenter as origin.
- `'local'` — pass `[x_km, y_km, z_km]` directly; no conversion is applied.

### Building the subsource list from a CSV

A common workflow is to load a kinematic rupture model from a CSV file:

```python
import pandas as pd

df = pd.read_csv('fault_model.csv', sep=';')
df = df[df['slip'] > 0].reset_index(drop=True)  # drop zero-slip patches

subsource_list = [
    {
        'centroid': [row['lon'], row['lat'], row['depth'] / 1000.],  # m → km
        'slip'    :  row['slip'],      # m
        'rupt_t'  :  row['rupture'],   # s
        'dip'     :  row['dip'],       # deg
        'rake'    :  row['rake'],      # deg
    }
    for _, row in df.iterrows()
]
```

A fully annotated version of this workflow, including result plotting and sanity checks, is provided in `examples/run_finite_fault.py`.


---

## Unit conventions

| Quantity       | Unit       |
|----------------|------------|
| Velocities     | km/s       |
| Depths/distances | km       |
| Seismic moment | dyne·cm    |
| Stress drop    | bar        |
| Frequency      | Hz         |
| Time           | s          |
| Acceleration   | cm/s²      |

---

## References

- Boore D.M. (2003). Simulation of ground motion using the stochastic method.
  *Pure Appl. Geophys.* 160, 635–676.
- Brune J.N. (1970). Tectonic stress and spectra of seismic shear waves.
  *J. Geophys. Res.* 75, 4997–5009.
- Motazedian D. & Atkinson G.M. (2005). Stochastic finite-fault modelling
  based on a dynamic corner frequency. *Bull. Seismol. Soc. Am.* 95, 995–1010.
- Dang et al. (2021). Updated corner-frequency model for stochastic
  finite-fault ground-motion simulation.
  *Bull. Seismol. Soc. Am.* 112, 921–938.
- Otarola C. & Ruiz S. (2016). Stochastic generation of accelerograms for subduction earthquakes.
  *Bull. Seismol. Soc. Am.* 106(6), 2511–2520. https://doi.org/10.1785/0120150262
- Ruiz S., Ojeda J., Pastén C., Otarola C. & Silva R. (2018). Stochastic strong-motion simulation in
  borehole and on surface for the 2011 Mw 9.0 Tohoku-Oki megathrust earthquake considering P, SV, and SH
  amplification transfer functions. *Bull. Seismol. Soc. Am.* 108(5A), 2333–2346.
  https://doi.org/10.1785/0120170342

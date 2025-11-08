# energy-plus-utility

Utilities and helpers for running [EnergyPlus](https://energyplus.net/) from Python notebooks (Colab-friendly) via `pyenergyplus`.  
Includes a **silent Colab bootstrapper** that installs system libraries, fetches EnergyPlus 25.1, and wires env/paths so `pyenergyplus` imports cleanly.

---

## Table of Contents

1. [Features](#-features)
2. [Package Layout](#-package-layout)
3. [Supported](#-supported)
4. [Quick Start (Colab)](#-quick-start-colab)
5. [Local (non-Colab) Setup](#-local-non-colab-setup)
6. [Quick Usage](#-quick-usage)
7. [Unified Runtime Hooks (Register at Runtime)](#-unified-runtime-hooks-register-at-runtime)
8. [Live Control & Monitoring Helpers](#-live-control--monitoring-helpers-actuators-variables-meters)
9. [Kalman/EKF: Persistent Per-Zone Estimation](#-kalmaneqkf-persistent-per-zone-estimation-pluggable)
10. [Minimal Run Sequence (with Callbacks & EKF)](#-minimal-run-sequence-with-callbacks--ekf)
11. [API Highlights](#-api-highlights-eplusutil)
12. [Troubleshooting](#-troubleshooting)
13. [SQL Explorer](#-sql-explorer-inspect--extract-from-eplusoutsql)
14. [License](#-license)
15. [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- One-line **Colab bootstrap**: prepares apt packages, installs `libssl1.1`, downloads EnergyPlus 25.1, sets `ENERGYPLUSDIR` and `LD_LIBRARY_PATH`, and updates `sys.path` (no prints).
- A high-level `EPlusUtil` class with:
  - A **unified runtime callback registry** — register/enable/disable/list/unregister handlers **at any runtime hook** (no subclassing).
  - Built-in handlers:
    - `probe_zone_air_and_supply` — fast per-zone snapshot of air state + supply aggregates (optionally logged).
    - `probe_zone_air_and_supply_with_kf` — **persistent Kalman/EKF** with pluggable model + fast SQLite logging.
    - CO₂ helpers (outdoor setpoint actuation), CSV-driven occupancy, HVAC “kill switch”, etc.
  - Robust discovery: safely **list variables/meters/actuators** without relying on brittle RDD/MDD/EDD files (uses dictionaries when present; falls back to API probing).
  - Ensure/patch **Output:SQLite**, **Output:Variable**, **Output:Meter**.
  - Query/plot series directly from **`eplusout.sql`** (variables & meters, resampling, unit conversion).
  - Weather extraction to CSV, covariance/correlation heatmaps, and more.

---

## 🧩 Package Layout

```
energy-plus-utility/
├─ pyproject.toml
├─ README.md  ← you’re reading this
└─ eplus/
   ├─ __init__.py            # exposes prepare_colab_eplus (lazy loads EPlusUtil)
   ├─ colab_bootstrap.py     # silent Colab runtime prep
   └─ eplus_util.py          # the EPlusUtil class
```

---

## 🐍 Supported

- Python 3.9–3.12
- Ubuntu 20.04/22.04 (Google Colab default is fine)
- EnergyPlus **25.1.0** (downloaded by the bootstrap)

---

## 🚀 Quick Start (Colab)

> Replace `dev` with a tag when you cut one.

### Option A — Python API (silent bootstrap)

```python
%pip install -q "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"

from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()  # installs deps, fetches E+, sets env/paths (silent)

from eplus.eplus_util import EPlusUtil
util = EPlusUtil(verbose=1)
```

### Option B — CLI helper (same bootstrap)

```python
%pip install -q "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"
!eplus-prepare-colab    # add --verbose to see logs

from eplus.eplus_util import EPlusUtil
util = EPlusUtil(verbose=1)
```

> **Important:** We lazy-load `EPlusUtil`. Always run `prepare_colab_eplus()` **before** importing `EPlusUtil` (if importing from `eplus.__init__`). Importing from `eplus.eplus_util` after bootstrap is always safe.

---

## 🔧 Local (non-Colab) Setup

If you already have EnergyPlus installed locally:

```bash
export ENERGYPLUSDIR="/path/to/EnergyPlus-25-1-0"
export LD_LIBRARY_PATH="$ENERGYPLUSDIR:$LD_LIBRARY_PATH"
# ensure EnergyPlus' Python site-packages (pyenergyplus) is importable
```

Then:

```python
pip install "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"
```

---

## 🧪 Quick Usage

### 1) Minimal run and SQL output

```python
from eplus.eplus_util import EPlusUtil

util = EPlusUtil(verbose=1, out_dir="eplus_out")
util.set_model(idf="/content/model.idf", epw="/content/weather.epw", out_dir="eplus_out")

# Ensure SQL is enabled, then run a design day
util.ensure_output_sqlite()
util.run_design_day()

# Plot a meter (auto-converts J → kWh)
util.plot_sql_meters(["Electricity:Facility"], reporting_freq=("TimeStep","Hourly"), resample="1H")
```

### 2) Add variables/meters programmatically

```python
# Add Zone Air Temperature for all zones (hourly), and a meter
util.ensure_output_variables([
    {"name": "Zone Air Temperature", "key": "*", "freq": "Hourly"},
])
util.ensure_output_meters(["Electricity:Facility"], freq="TimeStep")
util.run_annual()
```

### 3) Explore what’s available

```python
# Variables + meters (dictionary/API fallbacks; writes CSVs)
rows = util.list_variables_safely(kinds=("var","meter"))
# Actuators discovered via API-only catalog path (handles verified)
acts = util.list_actuators_safely()
zones = util.list_zone_names(save_csv=True)     # writes zones.csv into out_dir
```

### 4) Weather to CSV

```python
csv_path, summary = util.export_weather_sql_to_csv(resample="1H")
csv_path
```

### 5) CSV-driven occupancy

```python
# CSV must have a time column (e.g., 'timestamp') and zone-named columns
util.enable_csv_occupancy("/content/occ_schedule.csv", fill="ffill")
util.run_design_day()
```

### 6) CO₂ prep + outdoor setpoint actuation

```python
util.prepare_run_with_co2(outdoor_co2_ppm=420.0)
util.register_handlers("begin", [
  {"method_name":"co2_set_outdoor_ppm", "kwargs":{"value_ppm": 450}}
])
util.run_design_day()
```

---

## 🔁 Unified Runtime Hooks (Register at Runtime)

EnergyPlus exposes many hook points in the runtime API. `EPlusUtil` wraps them behind a **single registry** so you can attach your own Python methods **at runtime** — no subclassing required.

### Core methods

```python
register_handlers(hook, methods, *, clear=False, enable=True, run_during_warmup=None) -> list[str]
list_handlers(hook) -> list[str]
unregister_handlers(hook, names: list[str]) -> list[str]
enable_hook(hook) -> None
disable_hook(hook) -> None
```

- **`hook`** can be:
  - An **alias string**:  
    `"begin"`, `"before_hvac"`, `"inside_iter"`, `"after_hvac"`, `"after_zone"`, `"after_warmup"`, `"after_get_input"`
  - A **full runtime attribute name** (string), e.g.  
    `"callback_after_predictor_before_hvac_managers"`
  - The **registration callable itself**, e.g.  
    `util.api.runtime.callback_inside_system_iteration_loop`

- **`methods`** can be:
  - `["handler_name", "another_handler"]`, or
  - `[{"method_name": "handler_name", "kwargs": {...}}, ...]`  
    (kwargs key aliases accepted: `"kwargs"`, `"params"`, `"key_kwargs"`, `"key_wargs"`)

- **Handler signature:**  
  `handler(self, state, **kwargs)`

- **Ordering & de-dupe:**  
  Existing order is preserved; new names append. Re-registering a name updates its kwargs (last registration wins) without duplication.

- **Warmup control:**  
  `run_during_warmup=True|False|None` (default: skip during warmup).

- **Enable/disable:**  
  Temporarily toggle dispatch at any hook without altering the registered handlers.

### Hook alias map

| Alias             | EnergyPlus runtime registration method                         | Typical phase                                      |
|-------------------|----------------------------------------------------------------|----------------------------------------------------|
| `begin`           | `callback_begin_system_timestep_before_predictor`              | Start of each system timestep (“before predictor”) |
| `before_hvac`     | `callback_after_predictor_before_hvac_managers`                | After predictor, before HVAC managers              |
| `inside_iter`     | `callback_inside_system_iteration_loop`                        | Inside system iteration loop                       |
| `after_hvac`      | `callback_end_system_timestep_after_hvac_reporting`            | End of system timestep (after HVAC reporting)      |
| `after_zone`      | `callback_end_zone_timestep_after_zone_reporting`              | End of zone timestep                               |
| `after_warmup`    | `callback_after_new_environment_warmup_complete`               | After warmup complete                              |
| `after_get_input` | `callback_after_component_get_input`                           | After component input is read                      |

> You can also pass the **exact** `api.runtime.callback_*` function or its **string name**.

### Examples

#### A) Log zone state each system timestep (begin)

```python
util.register_handlers("begin", [
  {"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": 1}}
])
util.run_design_day()
```

#### B) After-HVAC end-of-timestep analytics

```python
util.register_handlers("after_hvac", [
  {"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": None}}
])
util.run_annual()
```

#### C) Update a handler’s kwargs without changing order

```python
# Initial
util.register_handlers("begin", [
  {"method_name": "co2_set_outdoor_ppm", "kwargs": {"value_ppm": 450}}
])

# Later: only update kwargs (last one wins)
util.register_handlers("begin", [
  {"method_name": "co2_set_outdoor_ppm", "kwargs": {"value_ppm": 500}}
])
```

#### D) Temporarily pause a hook

```python
util.disable_hook("begin")
util.run_design_day()  # handlers at "begin" won't run
util.enable_hook("begin")
```

#### E) Remove specific handlers

```python
util.unregister_handlers("begin", ["co2_set_outdoor_ppm"])
util.list_handlers("begin")
```

---

## 🔧 Live Control & Monitoring Helpers (Actuators, Variables, Meters)

These helpers are designed to be **re-usable** in your own controllers (the functions you register with `register_handlers(...)`). They resolve and cache handles **per run**, are warmup-aware, and can optionally log with timestamps.

### Actuators

**Get at runtime**
```python
val = util.runtime_get_actuator(
    s,
    component_type="Schedule:Compact",
    control_type="Schedule Value",
    actuator_key="FanAvailSched",
    allow_warmup=False,         # skip during warmup
    default=None                # return None if not available yet
)
```

**Set at runtime (returns True/False)**
```python
ok = util.runtime_set_actuator(
    s,
    component_type="People",
    control_type="Number of People",
    actuator_key="OpenOffice People",
    value=10.0,                 # or a callable: value=lambda self,s: ...
    clamp=(0.0, 100.0),         # optional
    allow_warmup=False,
    log=True
)
```

**Log an actuator each tick**  
(Use as a registered handler; supports `when="always"|"on_change"|"on_resolve"` and `include_timestamp=True`)
```python
util.register_handlers("begin", [{
  "method_name": "tick_log_actuator",
  "kwargs": {
    "component_type": "Weather Data",
    "control_type": "Outdoor Dry Bulb",
    "actuator_key": "Environment",
    "when": "on_resolve",
    "include_timestamp": True,
    "precision": 2
  }
}])
```

> **Mass flow setpoint example:** If your model exposes an actuator like  
> `ComponentType="System Node Setpoint", ControlType="Mass Flow Rate Setpoint", ActuatorKey="<inlet node>"`,  
> you can drive supply flow per zone by setting that actuator every timestep.

### Variables

**Get at runtime**
```python
tz = util.runtime_get_variable(
    s,
    name="Zone Air Temperature",
    key="SPACE1-1",
    allow_warmup=False,
    default=None
)
```

**Log a Zone People count (only on change)**
```python
util.register_handlers("begin", [{
  "method_name": "tick_log_variable",
  "kwargs": {
    "name": "Zone People Occupant Count",
    "key": "SPACE1-1",
    "when": "on_change",
    "precision": 0,
    "include_timestamp": True,
    "allow_warmup": False
  }
}])
```

> Register for many zones by creating one handler per zone key.

### Meters

**Get at runtime**
```python
p_el = util.runtime_get_meter(
    s,
    name="Electricity:Facility",
    allow_warmup=False,
    default=None
)
```

**Log a meter value each tick**
```python
util.register_handlers("after_hvac", [{
  "method_name": "tick_log_meter",
  "kwargs": {
    "name": "Electricity:Facility",
    "when": "on_change",            # or "always" / "on_resolve"
    "precision": 3,
    "include_timestamp": True,
    "allow_warmup": False
  }
}])
```

---

## 📈 Kalman/EKF: Persistent Per-Zone Estimation (Pluggable)

`probe_zone_air_and_supply_with_kf` layers a **Kalman/Extended Kalman filter** on top of the fast probe and persists estimates to SQLite.

### Measurement policy (robust by design)

- Outdoor & per-zone **air** (T, w, CO₂) with **forward-fill** at the measurement layer.
- Zone `w` fallback: payload → **Zone Mean Air Humidity Ratio** → derived from `(T, RH, P_site)` using Tetens.
- Supply aggregates via inlet nodes: mass flow, T, w, CO₂ (mass-flow-weighted).

### Pluggable model (“preparer”)

Provide:

```python
kf_prepare_fn(self?, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R) -> dict
# returns: {x_prev, P_prev, f_x, F, H, Q, R, y}
```

- Default: `_kf_prepare_inputs_zone_energy_model` — a practical thermal/moisture/CO₂ model using outdoor/supply deltas and a random-walk flavor for latent states.

### Persistence to SQLite (fast, batched)

- Default DB file: `out_dir/eplusout.sql` (coexists with EnergyPlus tables), or set `kf_db_filename="kalman.sqlite"`.
- Default table: `KalmanEstimates` with columns for measured `y_*`, predicted `yhat_*`, and **state vector** (`mu_*`).  
  Column names for state can be user-provided (`kf_state_col_names`); schema mutates safely on first write.

### One-liner

```python
util.register_handlers("begin", [
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {"log_every_minutes": None, "kf_log": True}}
])
util.run_annual()
```

### Configure noise, priors, and zones

```python
util.register_handlers("begin", [
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {
     "kf_sigma_P_diag": [1e-6, 5e-4, 1e-6, 1e-6, 5e-5],
     "kf_sigma_R_diag": [0.25**2, (3e-4)**2, 20.0**2],
     "kf_init_mu":      [0.0, 21.0, 0.0, 0.008, 420.0],
     "kf_init_cov_diag":[1.0, 25.0, 1.0, 1e-3, 1e3],
     "kf_zones": ["LIVING", "KITCHEN"],
     "kf_exclude_patterns": ("PLENUM",),
     "kf_db_filename": "kalman.sqlite",
     "kf_sql_table": "ZoneEKF",
     "kf_log": True
   }}
])
util.run_design_day()
```

### Bring your own model (custom preparer)

```python
def my_prepare(self, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R):
    import numpy as np
    n = len(mu_prev)
    F = np.eye(n); Q = Sigma_P
    H = np.zeros((3, n)); H[:,:3] = np.eye(3)  # observe first 3 states directly
    R = Sigma_R
    def f_x(x): return x  # random walk
    return dict(x_prev=mu_prev, P_prev=P_prev, f_x=f_x, F=F, H=H, Q=Q, R=R, y=meas["y"])

util.register_handlers("begin", [
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {"kf_prepare_fn": my_prepare, "kf_log": True}}
])
util.run_annual()
```

---

## 🔁 Minimal Run Sequence (with Callbacks & EKF)

**What’s required vs optional**

- **Required**
  - `set_model(...)` — define `idf`, `epw`, `out_dir`.
  - `register_handlers("begin", [...])` — attach runtime callbacks (e.g., EKF probe).
  - `run_design_day()` or `run_annual()` — actually run the simulation.

- **Optional / situational**
  - `ensure_output_variables([...])` / `ensure_output_meters([...])` — only if you want those series saved to `eplusout.sql`.
  - `ensure_output_sqlite()` — only if you need SQL outputs for post-run analysis/plots.
  - `delete_out_dir()` — only if you want a fully clean folder (runs already clear stale SQL/ERR/AUDIT files).
  - `enable_runtime_logging()` — helpful for debugging; not required.
  - Manual `new_state()` — not needed; `EPlusUtil` manages the state internally.

### Minimal example (EKF callback + optional SQL outputs)

```python
idf = f"{EPLUS}/ExampleFiles/5ZoneAirCooled.idf"
epw = f"{EPLUS}/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

util = EPlusUtil(verbose=1, out_dir="eplus_out")
util.set_model(
    idf, epw,
    outdoor_co2_ppm=400.0,          # optional (CO₂ helper)
    per_person_m3ps_per_W=3.82e-8   # optional (CO₂ helper)
)

# (Optional) ensure SQL time series exist for post-run analysis
util.ensure_output_variables([
    {"name": "Zone Mean Air Temperature",            "key": "*",           "freq": "TimeStep"},
    {"name": "Zone Mean Air Humidity Ratio",         "key": "*",           "freq": "TimeStep"},
    {"name": "Zone Air CO2 Concentration",           "key": "*",           "freq": "TimeStep"},
    {"name": "Site Outdoor Air Drybulb Temperature", "key": "Environment", "freq": "TimeStep"},
    {"name": "Site Outdoor Air Humidity Ratio",      "key": "Environment", "freq": "TimeStep"},
    {"name": "Site Outdoor Air Barometric Pressure", "key": "Environment", "freq": "TimeStep"},
    {"name": "System Node Temperature",              "key": "*",           "freq": "TimeStep"},
    {"name": "System Node Mass Flow Rate",           "key": "*",           "freq": "TimeStep"},
    {"name": "System Node Humidity Ratio",           "key": "*",           "freq": "TimeStep"},
])
util.ensure_output_meters([
    "InteriorLights:Electricity:Zone:SPACE5-1",
    "Cooling:EnergyTransfer:Zone:SPACE1-1",
    "Cooling:EnergyTransfer",
    "Electricity:Facility",
    "ElectricityPurchased:Facility",
    "ElectricitySurplusSold:Facility",
], freq="TimeStep")
util.ensure_output_sqlite()  # produce eplusout.sql

# ✅ The important part: register a runtime callback that runs every iteration
util.register_handlers("begin", [
    {"method_name": "probe_zone_air_and_supply_with_kf",
     "kwargs": {
         "log_every_minutes": 15,
         "precision": 3,

         # EKF persistence to SQLite (separate from eplusout.sql if desired)
         "kf_db_filename": "eplusout_kf_test.sqlite",
         "kf_batch_size": 50,
         "kf_commit_every_batches": 10,
         "kf_checkpoint_every_commits": 5,
         "kf_journal_mode": "WAL",
         "kf_synchronous": "NORMAL",

         # --- 10-state EKF init: (αo, αs, αe, βo, βs, βe, γe, Tz, wz, cz)
         "kf_init_mu":        [0.1, 0.1, 0.0,  0.1, 0.1, 0.0,  0.0,  20.0, 0.008, 400.0],
         "kf_init_cov_diag":  [1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0,  25.0, 1e-3,  1e3  ],
         "kf_sigma_P_diag":   [1e-6,1e-6,1e-6, 1e-6,1e-6,1e-6, 1e-6, 1e-5, 1e-6,  1e-4],

         # Optional: pretty column names for state persistence (dynamic schema)
         "kf_state_col_names": [
             "alpha_o","alpha_s","alpha_e","beta_o","beta_s","beta_e","gamma_e","Tz","wz","cz"
         ],

         # Use the 10-state preparer (pluggable)
         "kf_prepare_fn": util._kf_prepare_inputs_zone_energy_model
     }}
], run_during_warmup=False)

# Run the simulation (design-day or annual)
rc = util.run_annual()
```

**Why callbacks at runtime?**  
They let you inject logic **during** the simulation:
- Live probing of zone/supply conditions (T, w, CO₂, flows).
- On-the-fly estimation (KF/EKF) with durable, batched logging.
- Control/actuation logic (e.g., schedule overrides, People actuators) if you register such handlers.

> The EKF itself **does not** require `eplusout.sql`. SQL is only needed for post-run analysis/plots if you use those helpers.

---

## 📚 API Highlights (`EPlusUtil`)

**Model / Run**
- `set_model(...)`, `run_design_day()`, `run_annual()`, `dry_run_min(...)`, `set_simulation_params(...)`

**Unified Callbacks (runtime registry)**
- `register_handlers(hook, methods, *, clear=False, enable=True, run_during_warmup=None)`
- `list_handlers(hook)`, `unregister_handlers(hook, names)`
- `enable_hook(hook)`, `disable_hook(hook)`

**Discovery**
- `list_variables_safely(...)`, `list_actuators_safely(...)`, `list_zone_names(...)`

**Outputs / SQL**
- `ensure_output_sqlite()`, `ensure_output_variables([...])`, `ensure_output_meters([...])`
- `get_sql_series_dataframe([...])`, `plot_sql_series([...])`, `plot_sql_meters([...])`, `plot_sql_zone_variable(...)`

**Weather & Stats**
- `export_weather_sql_to_csv(...)`, `plot_sql_cov_heatmap(control_sels, output_sels, ...)`

**Occupancy & HVAC**
- `enable_csv_occupancy(...)`, `enable_hvac_off_via_schedules([...])`

**CO₂**
- `prepare_run_with_co2(...)`, `co2_set_outdoor_ppm(...)`

**Probes & EKF**
- `probe_zone_air_and_supply(...)`, `probe_zone_air_and_supply_with_kf(...)`

**Live Control & Monitoring**
- `runtime_get_actuator(...)`, `runtime_set_actuator(...)`, `tick_log_actuator(...)`
- `runtime_get_variable(...)`, `tick_log_variable(...)`
- `runtime_get_meter(...)`, `tick_log_meter(...)`

---

## 🛟 Troubleshooting

**`ModuleNotFoundError: No module named 'pyenergyplus'`**  
You imported `EPlusUtil` **before** the bootstrap (which adds EnergyPlus to `sys.path`). Run:

```python
from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()
from eplus.eplus_util import EPlusUtil
```

**`energyplus: error while loading shared libraries: libssl.so.1.1`**  
Use the bootstrap. If you bypassed it in Colab, install `libssl1.1` manually.

**`eplusout.sql not found`**  
Enable SQLite, then run a sim:

```python
util.ensure_output_sqlite()
util.run_design_day()
```

**Callbacks not firing?**  
- Ensure you **register before the run**.
- Check `enable=True` (or `enable_hook(hook)`).
- If you need them during sizing/warmup, pass `run_during_warmup=True`.

**Write/permission errors in `out_dir`**  
Use a writable path (e.g., `out_dir="eplus_out"`). The class tests writability and fails early.

---

## 📁 SQL Explorer: inspect & extract from `eplusout.sql`

This package includes `EPlusSqlExplorer` to browse/search/extract from EnergyPlus’s `eplusout.sql` **without** `pyenergyplus`.

> Location: `eplus/sql_explorer.py`  
> Import: `from eplus import EPlusSqlExplorer`

### Quick start

```python
xp = EPlusSqlExplorer("eplus_out/eplusout.sql")
xp.list_tables()[:10]
xp.peek("ReportData", 5)
hits = xp.search_value("Electricity:Facility")
df = xp.auto_extract_series("Electricity:Facility", to_kwh=True)
df.head()
```

Save directly to CSV:

```python
xp.auto_extract_series("Electricity:Facility", to_kwh=True, csv_out="facility_kWh.csv")
```

**Tips**
- **Timestamps:** EnergyPlus reports hour as **end-of-interval** (1–24). The extractor shifts to **interval start** for plotting sanity.
- **Frequencies:** If you get no rows, broaden `freq_whitelist` or include design days.

---

## 📄 License

MIT License

Copyright (c) 2025 Mugalan

---

## ❤️ Acknowledgements

Built on the excellent [EnergyPlus](https://energyplus.net/) simulation engine and its Python API (`pyenergyplus`).


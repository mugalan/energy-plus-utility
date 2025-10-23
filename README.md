# energy-plus-utility

Utilities and helpers for running EnergyPlus from Python notebooks (Colab-friendly) via pyenergyplus.
Includes a silent Colab bootstrapper that installs system libraries, fetches EnergyPlus 25.1, and wires env/paths so pyenergyplus imports cleanly.

✨ Features
	•	One-line Colab bootstrap: prepares apt packages, installs libssl1.1, downloads EnergyPlus 25.1, sets ENERGYPLUSDIR and LD_LIBRARY_PATH, and updates sys.path (no prints).
	•	A high-level EPlusUtil class with:
	•	A runtime callback registry (no subclassing needed): register/enable/disable/clear handlers at any time prior to a run for:
	•	Begin of iteration (zone/system timestep) hooks
	•	After HVAC reporting hooks
	•	Built-in callbacks:
	•	probe_zone_air_and_supply (fast per-zone snapshot of air state + supply aggregates)
	•	probe_zone_air_and_supply_with_kf (persistent Kalman/EKF, pluggable model, fast SQL logging)
	•	CO₂ helpers, CSV-driven occupancy, HVAC “kill switch,” and more
	•	Safely list variables/meters/actuators (no fragile dependencies on RDD/MDD/EDD, with smart fallbacks)
	•	Ensure/patch Output:SQLite, Output:Variable, Output:Meter
	•	Query/plot series directly from eplusout.sql (variables & meters, resampling, unit conversion)
	•	Weather extraction to CSV, covariance/correlation heatmaps, etc.

⸻

# 🧩 Package layout

energy-plus-utility/
├─ pyproject.toml
├─ README.md  ← you’re reading this
└─ eplus/
   ├─ __init__.py            # exposes prepare_colab_eplus (lazy loads EPlusUtil)
   ├─ colab_bootstrap.py     # silent Colab runtime prep
   └─ eplus_util.py          # the EPlusUtil class


⸻

🐍 Supported
	•	Python 3.9–3.12
	•	Ubuntu 20.04/22.04 (Google Colab default is fine)
	•	EnergyPlus 25.1.0 (downloaded by the bootstrap)

⸻

# 🚀 Quick start (Colab)

Replace dev with a tag when you cut one.

## Option A — Python API (silent bootstrap)

%pip install -q "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"

from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()  # runs apt, libssl1.1, downloads E+, sets env/paths (no prints)

from eplus.eplus_util import EPlusUtil
util = EPlusUtil(verbose=1)

## Option B — CLI helper (same bootstrap)

%pip install -q "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"
!eplus-prepare-colab    # add --verbose to see logs

from eplus.eplus_util import EPlusUtil
util = EPlusUtil(verbose=1)

Important: We lazy-load EPlusUtil. Always run prepare_colab_eplus() before importing EPlusUtil (if importing from eplus.__init__). Importing from eplus.eplus_util after bootstrap is always safe.

⸻

# 🔧 Local (non-Colab) setup

If you already have EnergyPlus installed locally:

export ENERGYPLUSDIR="/path/to/EnergyPlus-25-1-0"
export LD_LIBRARY_PATH="$ENERGYPLUSDIR:$LD_LIBRARY_PATH"
**ensure EnergyPlus' Python site-packages (pyenergyplus) is importable**

Then:

pip install "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"


⸻

# 🧪 Quick usage

1) Minimal run and SQL output

from eplus.eplus_util import EPlusUtil

util = EPlusUtil(verbose=1, out_dir="eplus_out")
util.set_model(idf="/content/model.idf", epw="/content/weather.epw", out_dir="eplus_out")

## Ensure SQL is enabled, then run a design-day
util.ensure_output_sqlite()
util.run_design_day()

## Plot a meter (auto-converts J → kWh)
util.plot_sql_meters(["Electricity:Facility"], reporting_freq=("TimeStep","Hourly"), resample="1H")

2) Add variables/meters programmatically

## Add Zone Air Temperature for all zones (hourly), and a meter
util.ensure_output_variables([
    {"name": "Zone Air Temperature", "key": "*", "freq": "Hourly"},
])
util.ensure_output_meters(["Electricity:Facility"], freq="TimeStep")
util.run_annual()

3) Explore what’s available

vars_and_meters = util.list_variables_safely()  # robust, with RDD/MDD/API fallbacks
acts = util.list_actuators_safely()
zones = util.list_zone_names(save_csv=True)     # writes zones.csv into out_dir

4) Weather to CSV

csv_path, summary = util.export_weather_sql_to_csv(resample="1H")
csv_path


⸻

# 🔁 Runtime callbacks & event model (register at runtime)

EnergyPlus exposes multiple hook points in the runtime API. EPlusUtil wraps these with registries that you can modify at runtime (in Python) without subclassing:
	•	register_begin_iteration(methods, *, clear=False, enable=True, run_during_warmup=None)
	•	Handlers run at the beginning of each iteration (zone/system timestep).
	•	register_after_hvac_reporting(methods, *, clear=False, enable=True, run_during_warmup=None)
	•	Handlers run after HVAC reporting at the system timestep.

Both accept:
	•	["handler_name", "another_handler"] or
	•	[{"method_name": "handler_name", "kwargs": {...}}, ...]
	•	Aliases accepted for kwargs: key_wargs (typo tolerated), kwargs, key_kwargs, params.
	•	Handlers are called as: handler(self, state, **kwargs)

Key properties
	•	Hot-swap friendly: You can call register_* multiple times between runs; set clear=True to replace, or re-register a name to update its kwargs (last wins).
	•	Order preservation with de-dupe: Existing order is preserved; new names are appended. Re-registering a name updates its kwargs without duplicating it.
	•	Warmup control: run_during_warmup lets you run handlers during sizing/warmup if you want (default is skipped during warmup).
	•	Enable/disable: Toggle with enable=; you can list_* or unregister_* as needed.

Examples

A) Minimal: log zone state each timestep

util.register_begin_iteration([
  {"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": 1}}
])
util.run_design_day()

B) Add + remove at runtime (before next run)

## Add a logger and a CO2 outdoor setpoint actuator
util.register_begin_iteration([
  "my_logger",
  {"method_name": "co2_set_outdoor_ppm", "kwargs": {"value_ppm": 450}},
])

## Later, update CO2 setpoint without changing order (last wins for kwargs)
util.register_begin_iteration([
  {"method_name": "co2_set_outdoor_ppm", "kwargs": {"value_ppm": 500}},
])

## Disable handlers for a run:
util.register_begin_iteration([], enable=False)
util.run_design_day()

## Re-enable + clear to start fresh:
util.register_begin_iteration([], clear=True, enable=True)

C) After-HVAC reporting hook (system-level post-processing)

util.register_after_hvac_reporting([
  {"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": None}}
])
util.run_annual()

D) CSV-driven occupancy + HVAC kill switch combo

## Prepare convenience states
util.enable_csv_occupancy("/content/occ_schedule.csv", fill="ffill")
util.enable_hvac_off_via_schedules(["Always_On_Discrete"])

util.register_begin_iteration([
  "tick_csv_occupancy",    # updates People actuators from CSV
  "tick_hvac_kill"         # forces target availability schedules to zero
])
util.run_design_day()

Tip: You can test your registry without running a full annual sim by using run_design_day() or even dry_run_min() (for dictionary generation). For performance, turn off frequent prints via log_every_minutes=None.

⸻

# 📈 Kalman/EKF: persistent per-zone estimation (pluggable)

probe_zone_air_and_supply_with_kf layers a Kalman/Extended Kalman filter on top of the fast probe:
	•	Inputs (measurement policy):
	•	Outdoor & per-zone air (T, w, CO₂) with forward-fill.
	•	Supply aggregates via inlet nodes: mass flow, T, w, CO₂.
	•	Humidity ratio w falls back to: payload → Zone Mean Air Humidity Ratio → derived from (T, RH, P_site) using Tetens.
	•	Pluggable model (“preparer”):
	•	Provide kf_prepare_fn(self?, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R) -> dict
	•	Return EKF inputs {x_prev, P_prev, f_x, F, H, Q, R, y}.
	•	Default preparer (_kf_prepare_inputs_zone_energy_model) implements a practical random-walk style thermal/moisture/CO₂ model with regressors from outdoor/supply deltas.
	•	Persistence to SQLite (fast, batched):
	•	Default DB file: out_dir/eplusout.sql (coexists with EnergyPlus tables), or set kf_db_filename="kalman.sqlite"
	•	Table (default KalmanEstimates): columns for measured y_*, predicted yhat_*, and state vector mu_* (auto-adds columns on first insert; can provide names)

One-liner example

## Register the EKF probe (suppress frequent console prints from the raw probe)
util.register_begin_iteration([
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {"log_every_minutes": None, "kf_log": True}}
])
util.run_annual()

This will:
	•	Run the fast probe each timestep,
	•	Apply forward-fill/fallbacks for y = [T, w, CO₂],
	•	Build a simple regressor matrix from supply/outdoor,
	•	Call the preparer to assemble EKF inputs,
	•	Run an EKF update,
	•	Persist y, yhat, and mu to SQLite in batches.

Configure noise, priors, and zones

util.register_begin_iteration([
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {
     "kf_sigma_P_diag": [1e-6, 5e-4, 1e-6, 1e-6, 5e-5],  # process noise diag
     "kf_sigma_R_diag": [0.25**2, (3e-4)**2, 20.0**2],  # meas noise diag (T,w,CO2)
     "kf_init_mu":      [0.0, 21.0, 0.0, 0.008, 420.0], # prior mean
     "kf_init_cov_diag":[1.0, 25.0, 1.0, 1e-3, 1e3],    # prior covariance diag
     "kf_zones": ["LIVING", "KITCHEN"],                 # optional filter
     "kf_exclude_patterns": ("PLENUM",),                # default: filter plenums
     "kf_db_filename": "kalman.sqlite",                 # write to a dedicated file
     "kf_sql_table": "ZoneEKF",
     "kf_log": True
   }}
])
util.run_design_day()

Bring your own model (custom preparer)

def my_prepare(self, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R):
    import numpy as np
    # Observations: y = [T, w, CO2]
    # Simple random-walk: x_k = x_{k-1} + noise
    n = len(mu_prev)
    F = np.eye(n)
    Q = Sigma_P
    # Direct observation of first 3 states:
    H = np.zeros((3, n)); H[:,:3] = np.eye(3)
    R = Sigma_R
    def f_x(x): return x
    return dict(x_prev=mu_prev, P_prev=P_prev, f_x=f_x, F=F, H=H, Q=Q, R=R, y=meas["y"])

util.register_begin_iteration([
  {"method_name": "probe_zone_air_and_supply_with_kf",
   "kwargs": {"kf_prepare_fn": my_prepare, "kf_log": True}}
])
util.run_annual()

Read your estimates back

import os, sqlite3, pandas as pd
db = os.path.join(util.out_dir, "kalman.sqlite")  # or "eplusout.sql" if you used default
conn = sqlite3.connect(db)
df = pd.read_sql_query("SELECT * FROM ZoneEKF WHERE Zone='LIVING' ORDER BY Timestamp", conn)
conn.close()
df.head()

Performance & reliability knobs
	•	Batching: kf_batch_size (default 50), kf_commit_every_batches (default 10)
	•	SQLite pragmas: kf_journal_mode="WAL", kf_synchronous="NORMAL"
	•	Checkpoints: kf_checkpoint_every_commits (default 5)
	•	Graceful degrade: If SQLite errors occur, persistence disables itself (your simulation proceeds; probe payloads still available in memory).

⸻

## 📚 API highlights (EPlusUtil)
	•	Model/run
set_model(...), run_design_day(), run_annual(), dry_run_min(...), set_simulation_params(...)
	•	Callbacks (runtime registry)
register_begin_iteration([...]), register_after_hvac_reporting([...]), plus list_* / unregister_*
	•	Dictionary & discovery
list_variables_safely(...), list_actuators_safely(...), list_zone_names(...)
	•	Outputs / SQL
ensure_output_sqlite(), ensure_output_variables([...]), ensure_output_meters([...]),
get_sql_series_dataframe([...]), plot_sql_series([...]), plot_sql_meters([...]), plot_sql_zone_variable(...)
	•	Weather & stats
export_weather_sql_to_csv(...), plot_sql_cov_heatmap(control_sels, output_sels, ...)
	•	Occupancy & HVAC
enable_csv_occupancy(...), enable_hvac_off_via_schedules([...])
	•	CO₂
prepare_run_with_co2(...), co2_set_outdoor_ppm(...)
	•	Probes & EKF
probe_zone_air_and_supply(...), probe_zone_air_and_supply_with_kf(...)

⸻

# 🛟 Troubleshooting

ModuleNotFoundError: No module named 'pyenergyplus'
You imported EPlusUtil before the bootstrap (which adds EnergyPlus to sys.path). Run:

from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()
from eplus.eplus_util import EPlusUtil

energyplus: error while loading shared libraries: libssl.so.1.1
Use the bootstrap. If you bypassed it in Colab, install libssl1.1 manually.

eplusout.sql not found
Enable SQLite, then run a sim:

util.ensure_output_sqlite()
util.run_design_day()

Callbacks not firing?
Make sure you register before the run, and that enable=True. If you want them to run during sizing/warmup, set run_during_warmup=True.

Write/permission errors in out_dir
Use a writable path (e.g., out_dir="eplus_out"). The class tests writability and fails early.

⸻

# 📁 SQL Explorer: inspect & extract from eplusout.sql

This package includes EPlusSqlExplorer to browse/search/extract from EnergyPlus’s eplusout.sql without pyenergyplus.

Location: eplus/sql_explorer.py
Import: from eplus import EPlusSqlExplorer

Quick start

xp = EPlusSqlExplorer("eplus_out/eplusout.sql")
xp.list_tables()[:10]
xp.peek("ReportData", 5)
hits = xp.search_value("Electricity:Facility")
df = xp.auto_extract_series("Electricity:Facility", to_kwh=True)
df.head()

Save directly to CSV:

xp.auto_extract_series("Electricity:Facility", to_kwh=True, csv_out="facility_kWh.csv")

Tips:
	•	E+ hour is end-of-interval → extractor shifts to start for plotting sanity.
	•	If you get no rows, broaden freq_whitelist or include design days.

⸻

# 📄 License

MIT © Mugalan. See LICENSE.

⸻

# ❤️ Acknowledgements

Built on the excellent EnergyPlus engine and its Python API (pyenergyplus).
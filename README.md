here’s a drop-in `README.md` you can use. it’s written for your repo name `energy-plus-utility`, exposes the silent Colab bootstrap, and documents the main `EPlusUtil` flows.

---

# energy-plus-utility

Utilities and helpers for running [EnergyPlus](https://energyplus.net/) from Python notebooks (Colab-friendly) via `pyenergyplus`.
Includes a **silent Colab bootstrapper** that installs system libraries, fetches EnergyPlus 25.1, and wires env/paths so `pyenergyplus` imports cleanly.

## ✨ Features

* One-line **Colab bootstrap**: prepares apt packages, installs `libssl1.1`, downloads EnergyPlus 25.1, sets `ENERGYPLUSDIR` and `LD_LIBRARY_PATH`, and updates `sys.path` (no prints).
* A high-level `EPlusUtil` class that wraps common workflows:

  * Run **design-day** and **annual** simulations with centralized callback registration.
  * Safely **list variables/meters/actuators** without relying on fragile outputs.
  * Ensure/patch **Output:SQLite**, **Output:Variable**, **Output:Meter**.
  * Query/plot series directly from **`eplusout.sql`** (variables & meters, resampling, unit conversion).
  * **CO₂** convenience (`prepare_run_with_co2`, outdoor CO₂ schedule actuation).
  * **CSV-driven occupancy** → drives People actuators from a time series.
  * HVAC “kill switch” by forcing availability schedules to zero.
  * Weather extraction to CSV, covariance/correlation heatmaps, and more.

---

## 🧩 Package layout

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

* Python 3.9–3.12
* Ubuntu 20.04/22.04 (Google Colab default is fine)
* EnergyPlus **25.1.0** (downloaded by the bootstrap)

---

## 🚀 Quick start (Colab)

> The package is installable from your `dev` branch. Replace with a tag when you cut one.

### Option A — Python API (silent bootstrap)

```python
%pip install -q "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"

from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()  # runs apt, libssl1.1, downloads E+, sets env/paths (no prints)

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

> **Important:** We lazy-load `EPlusUtil`. Always run `prepare_colab_eplus()` **before** importing `EPlusUtil` (if you import from `eplus.__init__`). Importing from `eplus.eplus_util` after bootstrap is always safe.

---

## 🔧 Local (non-Colab) setup

If you already have EnergyPlus installed locally:

1. Set:

   ```bash
   export ENERGYPLUSDIR="/path/to/EnergyPlus-25-1-0"
   export LD_LIBRARY_PATH="$ENERGYPLUSDIR:$LD_LIBRARY_PATH"
   ```
2. Ensure the **parent** folder (that contains `pyenergyplus/`) is on `PYTHONPATH` or `sys.path`.

Then:

```bash
pip install "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"
```

---

## 🧪 Quick usage

### 1) Minimal run and SQL output

```python
from eplus.eplus_util import EPlusUtil

util = EPlusUtil(verbose=1, out_dir="eplus_out")
util.set_model(idf="/content/model.idf", epw="/content/weather.epw", out_dir="eplus_out")

# Ensure SQL is enabled, then run a design-day
util.ensure_output_sqlite()
util.run_design_day()

# Plot a meter (auto-converts J → kWh)
util.plot_sql_meters(["Electricity:Facility"], reporting_freq=("TimeStep","Hourly"), resample="1H")
```

### 2) Add variables/meters programmatically

```python
# Add Zone Air Temperature for all zones (hourly), and a couple of meters
util.ensure_output_variables([
    {"name": "Zone Air Temperature", "key": "*", "freq": "Hourly"},
])
util.ensure_output_meters(["Electricity:Facility"], freq="TimeStep")

util.run_annual()
```

### 3) Explore what’s available

```python
# Variables/meters discovered without brittle outputs
vars_and_meters = util.list_variables_safely()
acts = util.list_actuators_safely()
zones = util.list_zone_names(save_csv=True)
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
util.register_begin_iteration([{"method_name":"co2_set_outdoor_ppm", "key_wargs":{"value_ppm": 450}}])
util.run_design_day()
```

---

## 📚 API highlights (`EPlusUtil`)

* **Model/run**

  * `set_model(idf, epw, out_dir, *, reset=True, add_co2=True, outdoor_co2_ppm=420.0)`
  * `run_design_day()`, `run_annual()`, `dry_run_min(...)`
  * `set_simulation_params(...)`, `clear_patched_idf()`
* **Dictionary & discovery**

  * `list_variables_safely(...)`, `list_actuators_safely(...)`
  * `list_zone_names(...)`, `flatten_mtd(...)`
* **Outputs / SQL**

  * `ensure_output_sqlite()`, `ensure_output_variables([...])`, `ensure_output_meters([...])`
  * `get_sql_series_dataframe([...])`
  * `plot_sql_series([...])`, `plot_sql_meters([...])`, `plot_sql_zone_variable(...)`
  * `plot_sql_net_purchased_electricity(...)`
* **Weather & stats**

  * `export_weather_sql_to_csv(...)`
  * `plot_sql_cov_heatmap(control_sels, output_sels, ...)`
* **Occupancy & HVAC**

  * `enable_csv_occupancy(csv_path, ...)`, `disable_csv_occupancy()`
  * `enable_hvac_off_via_schedules([...])`, `disable_hvac_off_via_schedules()`
  * `register_begin_iteration([...])` / `list_begin_iteration()` / `unregister_begin_iteration([...])`
* **CO₂**

  * `prepare_run_with_co2(...)`, `co2_set_outdoor_ppm(...)`

---

## 🛟 Troubleshooting

**`ModuleNotFoundError: No module named 'pyenergyplus'`**
You imported `EPlusUtil` **before** the bootstrap (which adds EnergyPlus to `sys.path`).
Fix: run:

```python
from eplus.colab_bootstrap import prepare_colab_eplus
prepare_colab_eplus()
from eplus.eplus_util import EPlusUtil
```

Or use the lazy import from `eplus.__init__` *after* calling `prepare_colab_eplus()`.

---

**`energyplus: error while loading shared libraries: libssl.so.1.1`**
The bootstrap installs `libssl1.1`. If you bypassed it, install manually (Colab):

```bash
apt-get update -y
cd /tmp && for V in 1.1.1f-1ubuntu2.22 1.1.1f-1ubuntu2.21 1.1.1f-1ubuntu2.20 1.1.1f-1ubuntu2.19 1.1.1f-1ubuntu2; do \
  wget -q "http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_${V}_amd64.deb" -O libssl1.1.deb && \
  apt-get install -y ./libssl1.1.deb && break; done
```

---

**`eplusout.sql not found`**
You didn’t enable SQLite or the run failed. Call:

```python
util.ensure_output_sqlite()
util.run_design_day()
```

Then re-run your SQL queries/plots.

---

**Write/permission errors in `out_dir`**
Use a path you can write in (e.g., `out_dir="eplus_out"`). The class does a quick write test and will raise early.

---

## 🧱 Design notes

* We don’t declare `pyenergyplus` as a Python dependency—it ships with the EnergyPlus distribution. The bootstrap fetches EnergyPlus, then updates `sys.path` so `from pyenergyplus.api import EnergyPlusAPI` works in the same kernel.
* Imports are **lazy** in `eplus.__init__` so you can always import the module, run the bootstrap, then use `EPlusUtil`.

---

## 📦 Install variants

* **Branch**
  `pip install "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@dev"`

* **Tag (recommended for reproducibility)**

  ```
  pip install "energy-plus-utility @ git+https://github.com/mugalan/energy-plus-utility.git@v0.1.0"
  ```

---

## 🤝 Contributing

PRs welcome! Please:

1. Keep public APIs documented in this README.
2. Add small, focused examples for new features.
3. Use semantic commit messages (`feat:`, `fix:`, `chore:`).

---

## 📄 License

MIT © Mugalan. See `LICENSE`.

---

## ❤️ Acknowledgements

Built on top of the excellent [EnergyPlus](https://energyplus.net/) simulation engine and its Python API (`pyenergyplus`).


from __future__ import annotations

import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib
from typing import List, Dict, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3

from pyenergyplus.api import EnergyPlusAPI

# ---------- CSV logger specs ----------

class EPlusUtil:
    """
    Utility wrapper around pyenergyplus EnergyPlusAPI.

    • State is created at init and is resettable.
    • set_model(idf, epw, out_dir)
    • run_design_day() / run_annual() with centralized callback registration
    • list_variables_safely() / list_actuators_safely() via tiny design-day runs
    • set_simulation_params() patches timestep/runperiod (writes ...__patched.idf)

    SQL-focused plotting/IO:
      - plot_sql_series(...)
      - plot_sql_meters(...)
      - plot_sql_zone_variable(...)
      - plot_sql_net_purchased_electricity(...)
      - export_weather_sql_to_csv(...)

    Notes:
      - reporting_freq=None in plotters means "no frequency filter" (passes through any freq present).
    """

    # ---- Schedule types we probe for actuators ----
    _SCHEDULE_TYPES = (
        "Schedule:Compact",
        "Schedule:Constant",
        "Schedule:File",
        "Schedule:Year",
    )

    # ----- lifecycle -----
    def __init__(self, *, verbose: int = 1, out_dir: str | None =None):
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        # model paths
        self.idf: Optional[str] = None
        self.epw: Optional[str] = None
        self.out_dir: Optional[str] = out_dir

        # patched idf (if any)
        self._patched_idf_path: Optional[str] = None
        self._orig_idf_path: Optional[str] = None

        # occupancy (lazy-init fields)
        self._occ_enabled: bool = False
        self._occ_df = None
        self._zone_to_people: Dict[str, List[str]] = {}
        self._people_handles: Dict[str, List[int]] = {}
        self._occ_fill: str = "ffill"
        self._occ_verbose: bool = True
        self._occ_ready: bool = False

        # optional: user-added callbacks (registrar, func) pairs
        self._extra_callbacks: List[Tuple[callable, callable]] = []

        # logging + verbosity
        self.verbose: int = int(verbose)
        self._runtime_log_enabled: bool = False
        self._runtime_log_func = None

    def _assert_out_dir_writable(self):
        import os, tempfile, pathlib
        assert self.out_dir, "set_model(...) first."
        os.makedirs(self.out_dir, exist_ok=True)
        # quick write test
        tmp = pathlib.Path(self.out_dir) / ".write_test.tmp"
        with open(tmp, "wb") as f:
            f.write(b"ok")
        tmp.unlink(missing_ok=True)

    def clear_eplus_outputs(self, patterns: tuple[str, ...] = ("eplusout.*",)) -> None:
        """
        Remove common EnergyPlus outputs in out_dir, especially a stale/locked eplusout.sql.
        Safe to call before runs.
        """
        import glob, os
        assert self.out_dir, "set_model(...) first."
        for pat in patterns:
            for p in glob.glob(os.path.join(self.out_dir, pat)):
                try: os.remove(p)
                except IsADirectoryError: pass
                except FileNotFoundError: pass
                except PermissionError: pass  # leave it if OS blocks; at least we tried

    def delete_out_dir(self):
        """
        Delete the output directory (`self.out_dir`) and all of its contents, if it exists.

        The directory is removed recursively via `shutil.rmtree(..., ignore_errors=True)`.
        Missing directories or removal errors are silently ignored. This only affects the
        on-disk folder; the `self.out_dir` attribute is not modified.
        """        
        import shutil, os
        if self.out_dir and os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir, ignore_errors=True)

    # --- tiny logger ---
    def _log(self, level: int, msg: str):
        if self.verbose >= level:
            print(msg)

    # --- register callbacks ----

    # --- unified "begin-iteration" callback registry (with kwargs) ---

    def _init_begin_tick_registry(self):
        if not hasattr(self, "_begin_tick_enabled"):
            self._begin_tick_enabled: bool = True
            self._begin_tick_run_during_warmup: bool = False
            # names kept for simple listing
            self._begin_tick_names: list[str] = []
            # callable + kwargs specs (ordered)
            self._begin_tick_specs: list[tuple[callable, dict]] = []

    def _begin_tick_dispatcher(self, s):
        """Master dispatcher at the beginning of each system timestep."""
        if not getattr(self, "_begin_tick_run_during_warmup", False) and self.api.exchange.warmup_flag(s):
            return
        if not getattr(self, "_begin_tick_enabled", True):
            return
        # snapshot to avoid mutation issues mid-iteration
        specs = list(getattr(self, "_begin_tick_specs", []))
        for func, kw in specs:
            try:
                func(s, **kw)
            except Exception as e:
                try:
                    nm = getattr(func, "__name__", "handler")
                    self._log(1, f"[begin-tick] {nm} failed: {e}")
                except Exception:
                    pass

    def _ensure_begin_tick_registered(self):
        self._init_begin_tick_registry()
        pair = (self.api.runtime.callback_begin_system_timestep_before_predictor, self._begin_tick_dispatcher)
        if pair not in self._extra_callbacks:
            self._extra_callbacks.append(pair)
        if getattr(self, "state", None):
            try:
                self.api.runtime.callback_begin_system_timestep_before_predictor(self.state, self._begin_tick_dispatcher)
            except Exception:
                pass

    def register_begin_iteration(self, methods, *, clear: bool = False,
                                enable: bool = True,
                                run_during_warmup: bool | None = None):
        """
        Register handlers to run at the **beginning of each EnergyPlus system timestep**
        (the “before predictor” phase). Handlers are methods on this class and are
        invoked as **bound** callables with the signature:
        `handler(self, state, **kwargs)`.

        Accepted `methods` formats (order matters):
        • `Sequence[str]` — method names on this instance, e.g.
            `["occupancy_handler", "co2_set_outdoor_ppm"]`
        • `Sequence[dict]` — objects with a method name and optional kwargs, e.g.
            `[{"method_name": "occupancy_handler", "kwargs": {"lam": 25}},
            {"name": "co2_set_outdoor_ppm", "params": {"value_ppm": 450}}]`

        For convenience, the kwargs key may also be given as `"key_wargs"` (typo accepted)
        or `"key_kwargs"`; all are treated identically to `"kwargs"`/`"params"`.

        Behavior & ordering rules
        -------------------------
        - If `clear=True`, any previously registered handlers are removed before adding new ones.
        - Handlers are **de-duplicated by method name**; when a name appears multiple times,
        the **last occurrence wins** (its kwargs are kept).
        - The resulting order preserves any existing kept handlers, then appends new names
        in the order provided.
        - If `enable` is `False`, the registry is kept but **dispatch is disabled** until
        re-enabled (see `enable_begin_iteration()` / `disable_begin_iteration()`).
        - By default, handlers **do not run during warmup**. Set `run_during_warmup=True`
        to allow warmup execution, or `False` to force skip. `None` leaves the current
        registry setting unchanged.

        Side effects
        ------------
        Ensures the underlying EnergyPlus callback
        (`callback_begin_system_timestep_before_predictor`) is registered on the current
        state so handlers will be dispatched on the next run.

        Returns
        -------
        list[str]
            The **ordered list of registered method names** after this call.

        Raises
        ------
        TypeError
            If a method spec is not `str` or `dict`, or if provided kwargs are not a `dict`.
        ValueError
            If a dict spec is missing a method name or it is empty after stripping.
        AttributeError
            If a named method does not exist on this instance or is not callable.

        Examples
        --------
        Basic registration with defaults:

        >>> util.register_begin_iteration([
        ...     "occupancy_handler",
        ...     {"method_name": "co2_set_outdoor_ppm", "kwargs": {"value_ppm": 450.0, "log_every_minutes": None}},
        ... ])

        Replace any existing handlers and **do not** run during warmup:

        >>> util.register_begin_iteration(
        ...     [{"name": "occupancy_handler", "params": {"lam": 30, "min": 10, "max": 40}}],
        ...     clear=True, run_during_warmup=False
        ... )

        Keep the current set but temporarily disable dispatch:

        >>> util.register_begin_iteration([], enable=False)   # methods unchanged; dispatcher paused
        """

        self._init_begin_tick_registry()
        if clear:
            self._begin_tick_names = []
            self._begin_tick_specs = []

        def _extract(method_item):
            # returns (name: str, func: callable, kwargs: dict)
            if isinstance(method_item, str):
                name, kwargs = method_item.strip(), {}
            elif isinstance(method_item, dict):
                name = str(method_item.get("method_name") or method_item.get("name") or "").strip()
                # tolerate 'key_wargs' typo and a few aliases
                kwargs = (method_item.get("key_wargs")
                        or method_item.get("kwargs")
                        or method_item.get("key_kwargs")
                        or method_item.get("params")
                        or {})
                if not isinstance(kwargs, dict):
                    raise TypeError(f"kwargs for '{name}' must be a dict")
            else:
                raise TypeError(f"Unsupported method spec: {method_item!r}")

            if not name:
                raise ValueError(f"Invalid method name in spec: {method_item!r}")

            func = getattr(self, name, None)
            if func is None or not callable(func):
                raise AttributeError(f"No callable '{name}' found on {self.__class__.__name__}")
            return name, func, dict(kwargs)

        # dedupe by method name (last one wins for kwargs)
        seen = {nm: (fn, kw) for nm, (fn, kw) in zip(self._begin_tick_names, self._begin_tick_specs)}
        for item in methods:
            nm, fn, kw = _extract(item)
            seen[nm] = (fn, kw)

        # re-materialize ordered lists (preserve prior order, append new names at end)
        ordered = []
        for nm in self._begin_tick_names:
            if nm in seen and nm not in ordered:
                ordered.append(nm)
        for item in methods:
            nm = item if isinstance(item, str) else (item.get("method_name") or item.get("name"))
            nm = str(nm).strip()
            if nm and nm not in ordered:
                ordered.append(nm)

        self._begin_tick_names = ordered
        self._begin_tick_specs = [seen[nm] for nm in ordered]

        self._begin_tick_enabled = bool(enable)
        if run_during_warmup is not None:
            self._begin_tick_run_during_warmup = bool(run_during_warmup)

        self._ensure_begin_tick_registered()
        return list(self._begin_tick_names)

    def unregister_begin_iteration(self, method_names: Sequence[str] | None = None) -> list[str]:
        """Remove one/more handlers by name. If None, remove all."""
        self._init_begin_tick_registry()
        if method_names is None:
            self._begin_tick_names, self._begin_tick_specs = [], []
            return []
        remove = {str(n).strip() for n in method_names}
        new_names = [nm for nm in self._begin_tick_names if nm not in remove]
        self._begin_tick_names = new_names
        self._begin_tick_specs = [getattr(self, nm) for nm in new_names]  # will be fixed below
        # rebuild specs from names keeping stored kwargs if available
        name_to_spec = {nm: spec for nm, spec in zip(self._begin_tick_names, getattr(self, "_begin_tick_specs", []))}
        specs = []
        for nm in new_names:
            fn = getattr(self, nm)
            kw = name_to_spec.get(nm, (None, {}))[1]
            specs.append((fn, kw))
        self._begin_tick_specs = specs
        return list(self._begin_tick_names)

    def enable_begin_iteration(self):  self._init_begin_tick_registry(); self._begin_tick_enabled = True
    def disable_begin_iteration(self): self._init_begin_tick_registry(); self._begin_tick_enabled = False

    def list_begin_iteration(self) -> list[dict]:
        """Return the ordered list of registered handlers at begin iterationwith kwargs."""
        self._init_begin_tick_registry()
        out = []
        for nm, (fn, kw) in zip(self._begin_tick_names, self._begin_tick_specs):
            out.append({"method_name": nm, "kwargs": dict(kw)})
        return out


    # --- unified "after-HVAC-reporting" callback registry (with kwargs) ---

    def _init_after_hvac_registry(self):
        if not hasattr(self, "_after_hvac_enabled"):
            self._after_hvac_enabled: bool = True
            self._after_hvac_run_during_warmup: bool = False
            self._after_hvac_names: list[str] = []
            self._after_hvac_specs: list[tuple[callable, dict]] = []

    def _after_hvac_dispatcher(self, s):
        """Master dispatcher right AFTER HVAC reporting each system timestep (node data finalized)."""
        ex = self.api.exchange
        if not getattr(self, "_after_hvac_run_during_warmup", False) and ex.warmup_flag(s):
            return
        if not getattr(self, "_after_hvac_enabled", True):
            return
        # snapshot to avoid mutation during iteration
        specs = list(getattr(self, "_after_hvac_specs", []))
        for func, kw in specs:
            try:
                func(s, **kw)   # NOTE: bound method; signature def method(self, state, **kwargs)
            except Exception as e:
                try:
                    nm = getattr(func, "__name__", "handler")
                    self._log(1, f"[after-hvac] {nm} failed: {e}")
                except Exception:
                    pass

    def _ensure_after_hvac_registered(self):
        self._init_after_hvac_registry()
        if not hasattr(self, "_extra_callbacks"):
            self._extra_callbacks = []
        pair = (self.api.runtime.callback_end_system_timestep_after_hvac_reporting, self._after_hvac_dispatcher)
        if pair not in self._extra_callbacks:
            self._extra_callbacks.append(pair)
        if getattr(self, "state", None):
            try:
                self.api.runtime.callback_end_system_timestep_after_hvac_reporting(self.state, self._after_hvac_dispatcher)
            except Exception:
                pass

    def register_after_hvac_reporting(self, methods, *, clear: bool = False,
                                    enable: bool = True,
                                    run_during_warmup: bool | None = None) -> list[str]:
        """
        Register handlers to run **after HVAC reporting** at each system timestep
        (i.e., right after node data are finalized). Handlers must be methods on
        this class and are invoked as **bound** callables with the signature:
        `handler(self, state, **kwargs)`.

        Accepted `methods` formats (order matters):
        • `Sequence[str]` — method names on this instance, e.g.
            `["probe_zone_air_and_supply", "my_logger"]`
        • `Sequence[dict]` — objects with a method name and optional kwargs, e.g.
            `[{"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": 5}},
            {"name": "my_logger", "params": {"level": "debug"}}]`

        Notes on kwargs keys:
        You may use `"kwargs"`, `"params"`, `"key_kwargs"`, or even the tolerated typo
        `"key_wargs"`—they are all treated equivalently.

        Behavior & ordering rules
        -------------------------
        - If `clear=True`, previously registered handlers are removed before adding new ones.
        - Handlers are **de-duplicated by method name**; the **last occurrence wins** (its
        kwargs override prior ones).
        - Final order preserves surviving existing handlers, then appends any new names in
        the order provided.
        - If `enable` is `False`, the registry is kept but **dispatch is disabled** until
        re-enabled (see `enable_after_hvac_reporting()` / `disable_after_hvac_reporting()`).
        - Warmup behavior: by default, handlers **do not run during warmup**. Pass
        `run_during_warmup=True` to enable, `False` to force-disable, or `None` to leave
        the existing setting unchanged.

        Side effects
        ------------
        Ensures the underlying EnergyPlus callback
        (`callback_end_system_timestep_after_hvac_reporting`) is registered on the current
        state so handlers will be dispatched on the next run.

        Returns
        -------
        list[str]
            The **ordered list of registered method names** after this call.

        Raises
        ------
        TypeError
            If a method spec is not `str` or `dict`, or if provided kwargs are not a `dict`.
        ValueError
            If a dict spec lacks a method name or it is empty after stripping.
        AttributeError
            If a named method does not exist on this instance or is not callable.

        Examples
        --------
        Register a probe and a logger with custom parameters:

        >>> util.register_after_hvac_reporting([
        ...     {"method_name": "probe_zone_air_and_supply", "kwargs": {"log_every_minutes": 1, "precision": 2}},
        ...     {"name": "my_logger", "params": {"tag": "after-hvac"}}
        ... ])

        Replace any existing handlers and keep them from running during warmup:

        >>> util.register_after_hvac_reporting(
        ...     ["probe_zone_air_and_supply"],
        ...     clear=True, run_during_warmup=False
        ... )

        Keep the current set but pause dispatch:

        >>> util.register_after_hvac_reporting([], enable=False)  # handlers unchanged; dispatcher paused
        """


        self._init_after_hvac_registry()
        if clear:
            self._after_hvac_names = []
            self._after_hvac_specs = []

        def _extract(method_item):
            # returns (name: str, func: callable, kwargs: dict)
            if isinstance(method_item, str):
                name, kwargs = method_item.strip(), {}
            elif isinstance(method_item, dict):
                name = str(method_item.get("method_name") or method_item.get("name") or "").strip()
                kwargs = (method_item.get("key_wargs")
                        or method_item.get("kwargs")
                        or method_item.get("key_kwargs")
                        or method_item.get("params")
                        or {})
                if not isinstance(kwargs, dict):
                    raise TypeError(f"kwargs for '{name}' must be a dict")
            else:
                raise TypeError(f"Unsupported method spec: {method_item!r}")

            if not name:
                raise ValueError(f"Invalid method name in spec: {method_item!r}")

            func = getattr(self, name, None)
            if func is None or not callable(func):
                raise AttributeError(f"No callable '{name}' found on {self.__class__.__name__}")
            return name, func, dict(kwargs)

        # de-dupe by method name (last wins)
        seen = {nm: (fn, kw) for nm, (fn, kw) in zip(self._after_hvac_names, self._after_hvac_specs)}
        for item in methods:
            nm, fn, kw = _extract(item)
            seen[nm] = (fn, kw)

        # rebuild ordered list: keep existing order, then append any new ones
        ordered = []
        for nm in self._after_hvac_names:
            if nm in seen and nm not in ordered:
                ordered.append(nm)
        for item in methods:
            nm = item if isinstance(item, str) else (item.get("method_name") or item.get("name"))
            nm = str(nm).strip()
            if nm and nm not in ordered:
                ordered.append(nm)

        self._after_hvac_names = ordered
        self._after_hvac_specs = [seen[nm] for nm in ordered]

        self._after_hvac_enabled = bool(enable)
        if run_during_warmup is not None:
            self._after_hvac_run_during_warmup = bool(run_during_warmup)

        self._ensure_after_hvac_registered()
        return list(self._after_hvac_names)

    def unregister_after_hvac_reporting(self, method_names: list[str] | None = None) -> list[str]:
        """Remove one/more handlers by name. If None, remove all."""
        self._init_after_hvac_registry()
        if method_names is None:
            self._after_hvac_names, self._after_hvac_specs = [], []
            return []
        remove = {str(n).strip() for n in method_names}
        # map current -> specs to preserve kwargs
        current = {nm: spec for nm, spec in zip(self._after_hvac_names, self._after_hvac_specs)}
        new_names = [nm for nm in self._after_hvac_names if nm not in remove]
        self._after_hvac_names = new_names
        self._after_hvac_specs = [current[nm] for nm in new_names]
        return list(self._after_hvac_names)

    def enable_after_hvac_reporting(self):  self._init_after_hvac_registry(); self._after_hvac_enabled = True
    def disable_after_hvac_reporting(self): self._init_after_hvac_registry(); self._after_hvac_enabled = False

    def list_after_hvac_reporting(self) -> list[dict]:
        """Return the ordered list of registered handlers with kwargs."""
        self._init_after_hvac_registry()
        out = []
        for nm, (fn, kw) in zip(self._after_hvac_names, self._after_hvac_specs):
            out.append({"method_name": nm, "kwargs": dict(kw)})
        return out

    # ---------- common SQL helpers ----------
    def _sql_minute_col(self, conn) -> str:
        """Detect the minute column name in the Time table ('Minute' vs 'Minutes')."""
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(Time)").fetchall()}
            if "Minute" in cols:
                return "Minute"
            if "Minutes" in cols:
                return "Minutes"
        except Exception:
            pass
        return "Minute"

    def _sql_freq_clause(self, freqs: Optional[Tuple[str, ...]]):
        """
        Build a tolerant SQL WHERE snippet + params for frequency filters.
        Returns (clause_str, params_list) where clause_str starts with 'AND ...' or ''.
        """
        if not freqs:
            return "", []
        mapping = {
            "TIMESTEP": [("%TIMESTEP%", "LIKE"), ("DETAILED", "EQ")],
            "HOURLY":   [("%HOUR%", "LIKE")],
            "DAILY":    [("%DAY%", "LIKE")],
            "MONTHLY":  [("%MONTH%", "LIKE")],
            "RUNPERIOD":[("%RUN%PERIOD%", "LIKE")],
        }
        conds, params = [], []
        for f in freqs:
            key = str(f).strip().upper()
            patterns = mapping.get(key, [(key, "EQ")])
            sub_conds = []
            for pat, mode in patterns:
                if mode == "LIKE":
                    sub_conds.append("UPPER(d.ReportingFrequency) LIKE ?")
                    params.append(pat)
                else:
                    sub_conds.append("UPPER(d.ReportingFrequency) = ?")
                    params.append(pat)
            conds.append("(" + " OR ".join(sub_conds) + ")")
        return ("AND (" + " OR ".join(conds) + ")") if conds else "", params

    def enable_runtime_logging(self):
        def _on_msg(msg):
            s = msg.decode("utf-8", errors="ignore") if isinstance(msg, (bytes, bytearray)) else str(msg)
            print(s.rstrip())
        self._runtime_log_enabled = True
        self._runtime_log_func = _on_msg
        self.api.runtime.callback_message(self.state, _on_msg)

    def disable_runtime_logging(self):
        self._runtime_log_enabled = False
        self._runtime_log_func = None
        # no unregister API; future resets will simply not re-register

    def _register_callbacks(self):
        """Register the single timestep callback (and any extras)."""
        # CSV occupancy tick (if enabled)
        if getattr(self, "_occ_enabled", False) and self._occ_df is not None:
            self.api.runtime.callback_begin_system_timestep_before_predictor(
                self.state, self._occ_cb_tick
            )

        # Any extra user-queued callbacks
        for registrar, func in getattr(self, "_extra_callbacks", []):
            registrar(self.state, func)

        # Re-attach runtime logger if enabled (EnergyPlus 25.1 uses 1-arg signature)
        if getattr(self, "_runtime_log_enabled", False) and getattr(self, "_runtime_log_func", None):
            self.api.runtime.callback_message(self.state, self._runtime_log_func)

    def reset_state(self) -> None:
        try:
            if getattr(self, "state", None):
                self.api.state_manager.reset_state(self.state)
        except Exception:
            pass
        self.state = self.api.state_manager.new_state()
        # occupancy runtime only
        self._people_handles = {}
        self._occ_ready = False
        # keep logger alive across resets
        if getattr(self, "_runtime_log_enabled", False) and getattr(self, "_runtime_log_func", None):
            self.api.runtime.callback_message(self.state, self._runtime_log_func)

    def set_model(self, idf: str, epw: str, out_dir: Optional[str] = None, *, reset: bool = True, add_co2: bool = True, outdoor_co2_ppm: float = 420.0,per_person_m3ps_per_W: float = 3.82e-8) -> None:
        """
        Configure the active EnergyPlus model paths and (optionally) inject a minimal
        CO₂ setup, ready for subsequent runs.

        This sets `self.idf`, `self.epw`, and `self.out_dir` (creating the output
        directory if needed). If `reset=True`, the EnergyPlus state is reset so that
        subsequent runs start clean.

        If `add_co2=True`, this calls `prepare_run_with_co2(...)` to:
        - enable zone CO₂ accounting via `ZoneAirContaminantBalance`,
        - create/bind an **outdoor CO₂ schedule** seeded to `outdoor_co2_ppm`,
        - patch each `People` object with a **CO₂ generation rate coefficient**
            (`per_person_m3ps_per_W`, in m³·s⁻¹ per W per person),
        - write a patched IDF in `out_dir` and switch `self.idf` to that file.
        (That helper also resets state by default, so the model will be ready to run
        with the CO₂ features active.)

        Parameters
        ----------
        idf : str
            Path to the IDF model to load.
        epw : str
            Path to the EPW weather file to use.
        out_dir : Optional[str], default None
            Directory for EnergyPlus outputs; created if missing. Defaults to
            ``"eplus_out"`` when not provided.
        reset : bool, default True
            If True, reset the EnergyPlus API state immediately after setting paths.
        add_co2 : bool, default True
            If True, inject the minimal CO₂ workflow via `prepare_run_with_co2(...)`
            and switch `self.idf` to the patched file.
        outdoor_co2_ppm : float, default 420.0
            Initial value for the outdoor CO₂ schedule (ppm) when `add_co2=True`.
        per_person_m3ps_per_W : float, default 3.82e-8
            People CO₂ generation coefficient (m³/s per W per person). EnergyPlus’s
            default is 3.82e-8; values are clamped to the model’s allowed range
            inside the helper.

        Notes
        -----
        - This method **does not run** a simulation; it only configures paths/state.
        - When `add_co2=True`, `self._orig_idf_path` is remembered and `self.idf`
        points to the newly written CO₂-patched IDF in `out_dir`.

        Returns
        -------
        None

        Examples
        --------
        Basic setup with CO₂ enabled (default):
        >>> util.set_model("models/small_office.idf", "weather/USA_CA_San-Francisco.epw",
        ...                out_dir="runs/run1")

        Custom outdoor CO₂ and generation rate:
        >>> util.set_model("bldg.idf", "site.epw", out_dir="out",
        ...                add_co2=True, outdoor_co2_ppm=450.0,
        ...                per_person_m3ps_per_W=3.5e-8)

        Skip CO₂ patching entirely:
        >>> util.set_model("bldg.idf", "site.epw", out_dir="out", add_co2=False)
        """

        self.idf = str(idf)
        self.epw = str(epw)
        self.out_dir = str(out_dir or "eplus_out")
        os.makedirs(self.out_dir, exist_ok=True)
        if reset:
            self.reset_state()
                
        if add_co2:
            self.prepare_run_with_co2(outdoor_co2_ppm=outdoor_co2_ppm,per_person_m3ps_per_W=per_person_m3ps_per_W)

    def list_zone_names(
        self,
        *,
        preferred_sources=("sql", "api", "idf"),
        save_csv: bool = False,
        csv_path: str | None = None,
    ) -> list[str]:
        """
        Return the list of **Zone** object names for the current model, probing one or
        more sources **in order** until names are found.

        Probe order & methods
        ---------------------
        - **"sql"**: Read `eplusout.sql` → `Zones` table (fastest; requires a prior run
        with `Output:SQLite` enabled). Falls back to a DISTINCT query if `ZoneIndex`
        isn’t present.
        - **"api"**: Perform a tiny **design-day** run in a temporary directory and query
        object names **after inputs are parsed** (uses a throwaway state; does not
        affect the main state or output folder).
        - **"idf"**: Regex-parse the active IDF for `Zone, ...;` blocks and take the first
        field as the zone name (comments removed first).

        The first source in `preferred_sources` that yields any names is used; later
        sources are skipped.

        Parameters
        ----------
        preferred_sources : Sequence[str], default ("sql", "api", "idf")
            Ordered list of sources to try. Any subset/ordering is allowed, e.g.
            `("idf",)` to avoid running EnergyPlus.
        save_csv : bool, default False
            If `True`, write a one-column CSV (`zone_name`) with the discovered names.
        csv_path : str | None, default None
            Path for the CSV when `save_csv=True`. Defaults to `<out_dir>/zones.csv`.

        Returns
        -------
        list[str]
            Ordered, de-duplicated list of zone names (whitespace-trimmed; empty/blank
            names filtered).

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called (requires `self.idf`
            and `self.out_dir`; the `"api"` path also requires `self.epw` to be set).

        Notes
        -----
        - The `"sql"` path is fastest when `eplusout.sql` already exists; otherwise the
        function transparently falls through to the next source.
        - The `"api"` probe runs EnergyPlus with `--design-day` in a temp dir using a
        **new state**, then immediately stops after inputs are parsed.
        - The `"idf"` parser is intentionally simple but robust enough for typical IDFs.

        Examples
        --------
        Basic usage (prefer SQL, then API, then IDF):

        >>> zones = util.list_zone_names()
        >>> zones[:3]
        ['CORE_ZN', 'PERIMETER_ZN_1', 'PERIMETER_ZN_2']

        Avoid any runs; parse only the IDF:

        >>> zones = util.list_zone_names(preferred_sources=("idf",))

        Try API first, then IDF; also save a CSV:

        >>> zones = util.list_zone_names(
        ...     preferred_sources=("api", "idf"),
        ...     save_csv=True,
        ...     csv_path="runs/zones.csv"
        ... )
        """
        assert self.idf and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        names: list[str] = []

        def _from_sql() -> list[str]:
            sql_path = os.path.join(self.out_dir, "eplusout.sql")
            if not os.path.exists(sql_path):
                return []
            try:
                conn = sqlite3.connect(sql_path)
                try:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(Zones)").fetchall()}
                    if {"ZoneIndex", "ZoneName"}.issubset(cols):
                        rows = conn.execute("SELECT ZoneName FROM Zones ORDER BY ZoneIndex").fetchall()
                    elif "ZoneName" in cols:
                        rows = conn.execute("SELECT DISTINCT ZoneName FROM Zones ORDER BY ZoneName").fetchall()
                    else:
                        return []
                    return [r[0] for r in rows if r and r[0]]
                finally:
                    conn.close()
            except Exception:
                return []

        def _from_api() -> list[str]:
            tmp_state = self.api.state_manager.new_state()
            bucket = {"zones": None}

            def after_get_input(s):
                z = self.api.exchange.get_object_names(s, "Zone") or []
                bucket["zones"] = list(z)
                try:
                    self.api.runtime.stop_simulation(s)
                except Exception:
                    pass

            with tempfile.TemporaryDirectory() as tdir:
                try:
                    self.api.runtime.callback_after_component_get_input(tmp_state, after_get_input)
                    self.api.runtime.run_energyplus(tmp_state, ['-w', self.epw, '-d', tdir, '--design-day', self.idf])
                finally:
                    self.api.state_manager.reset_state(tmp_state)
            return bucket["zones"] or []

        def _from_idf() -> list[str]:
            # Very simple parser: capture 'Zone, ... ;' blocks and take the first field as the name
            text = pathlib.Path(self.idf).read_text(errors="ignore")
            # Remove comments
            text_nc = re.sub(r'!.*$', '', text, flags=re.MULTILINE)
            # Find Zone objects
            blocks = re.findall(r'(?is)^\s*Zone\s*,(.*?);', text_nc, flags=re.MULTILINE)
            out = []
            for blk in blocks:
                parts = [p.strip().strip('"').strip("'") for p in blk.split(",")]
                if parts and parts[0]:
                    out.append(parts[0])
            seen, ordered = set(), []
            for n in out:
                if n not in seen:
                    seen.add(n); ordered.append(n)
            return ordered

        for src in preferred_sources:
            if names:
                break
            src = src.lower()
            if src == "sql":
                names = _from_sql()
            elif src == "api":
                names = _from_api()
            elif src == "idf":
                names = _from_idf()

        names = [n for n in names if str(n).strip()]
        if save_csv:
            path = csv_path or os.path.join(self.out_dir, "zones.csv")
            with open(path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["zone_name"])
                for n in names:
                    w.writerow([n])
            self._log(1, f"[zones] Wrote {path} (n={len(names)})")
        return names

    def prepare_run_with_co2(
        self,
        *,
        outdoor_co2_ppm: float = 420.0,
        schedule_name: str = "CO2-Outdoor-Actuated",
        # IMPORTANT: units are m3/s-W per person (per Watt of activity)
        per_person_m3ps_per_W: float = 3.82e-8,   # EnergyPlus default coefficient
        # Optional convenience: if you want to think in m3/s-person at a ref activity (W/person),
        target_per_person_m3ps_at_activity_W: float | None = None,
        target_activity_W_per_person: float = 100.0,   # only used with the option above
        wipe_outputs: bool = True,
        activate: bool = True,
        reset: bool = True,
    ) -> str:
        """
        Minimal, safe CO₂ prep for E+ 25.1:

        - Remove ALL Output:Variable (avoid legacy parse issues).
        - Enable CO₂ via ZoneAirContaminantBalance and bind an actuated Schedule:Constant (ppm).
        - Set People 'Carbon Dioxide Generation Rate' (NUMERIC, m3/s-W per person).
        - Does NOT add Output:SQLite/Meter/Variable.
        """
        assert self.idf and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        import pathlib, re, os

        # --- compute the coefficient safely (respect E+ limits) ---
        # E+ default: 3.82e-8 m3/s-W ; allowed up to 10x default
        DEFAULT_COEF = 3.82e-8
        MAX_COEF = 10.0 * DEFAULT_COEF

        coef = per_person_m3ps_per_W
        if target_per_person_m3ps_at_activity_W is not None:
            # Convert a per-person rate at a reference activity (W/person) to the per-W coefficient
            if target_activity_W_per_person <= 0:
                raise ValueError("target_activity_W_per_person must be > 0")
            coef = float(target_per_person_m3ps_at_activity_W) / float(target_activity_W_per_person)

        # clamp to allowed range
        coef = max(0.0, min(float(coef), MAX_COEF))

        src = pathlib.Path(self.idf)
        text = src.read_text(errors="ignore")

        def _append(s: str, block: str) -> str:
            s = s.rstrip() + "\n"
            return s + "\n" + block.strip() + "\n"

        def _rm_all_blocks(t: str, obj_pat: str) -> str:
            rx = re.compile(rf'(?is)^\s*{obj_pat}\s*,.*?;(?:[ \t]*\r?\n|$)', re.MULTILINE)
            return rx.sub("", t)

        # 0) Remove Output:Variable (robustness) and any legacy zone CO2 sources
        text = _rm_all_blocks(text, r'Output\s*:\s*Variable')
        text = _rm_all_blocks(text, r'ZoneContaminantSourceAndSink\s*:\s*CarbonDioxide')

        # 1) Minimal ScheduleTypeLimits
        if not re.search(r'(?im)^\s*ScheduleTypeLimits\s*,\s*Any\s+Number\s*[,;]', text):
            text = _append(text, "ScheduleTypeLimits,\n  Any Number;")

        # 2) Outdoor CO₂ schedule (ppm) for actuation
        text = _rm_all_blocks(text, rf'Schedule\s*:\s*Constant\s*,\s*(?:"{re.escape(schedule_name)}"|{re.escape(schedule_name)})')
        text = _append(text, f"""Schedule:Constant,
        {schedule_name},
        Any Number,
        {float(outdoor_co2_ppm):.6f};""")

        # 3) Enable CO₂ and bind outdoor schedule
        text = _rm_all_blocks(text, r'ZoneAirContaminantBalance')
        text = _append(text, f"""ZoneAirContaminantBalance,
        Yes,                 !- Carbon Dioxide Concentration
        {schedule_name},     !- Outdoor Carbon Dioxide Schedule Name
        No,                  !- Generic Contaminant Concentration
        ;                    !- Outdoor Generic Contaminant Schedule Name""")

        # 4) Patch People: set NUMERIC 'Carbon Dioxide Generation Rate' (m3/s-W)
        people_rx = re.compile(r'(?is)^\s*People\s*,\s*(?P<body>.*?);', re.MULTILINE)

        def _patch_people_block(body: str) -> str:
            import re
            lines = body.splitlines()

            label_num = "Carbon Dioxide Generation Rate"
            label_act = "Activity Level Schedule Name"

            def _has_label(line: str, label: str) -> bool:
                return re.search(rf'!\-\s*{re.escape(label)}\s*$', line, flags=re.I) is not None

            # Replace if present
            for i, ln in enumerate(lines):
                if _has_label(ln, label_num):
                    label_part = ln.split('!-', 1)[1].strip() if '!- ' in ln or '!- ' in ln else label_num
                    lines[i] = f"  {coef:.8g},  !- {label_part}"
                    return "\n".join(lines)

            # Else insert after Activity Level Schedule Name if available
            insert_at = None
            for i, ln in enumerate(lines):
                if _has_label(ln, label_act):
                    insert_at = i + 1
                    if not lines[i].rstrip().endswith(','):
                        lines[i] = lines[i].rstrip().rstrip(';') + ","
                    break

            new_line = f"  {coef:.8g},  !- {label_num}"

            if insert_at is None:
                # append near end — ensure previous line ends with a comma
                if lines:
                    if not lines[-1].rstrip().endswith(','):
                        lines[-1] = lines[-1].rstrip().rstrip(';') + ","
                    lines.append(new_line)
                else:
                    lines = [new_line]
            else:
                lines.insert(insert_at, new_line)

            return "\n".join(lines)

        def _people_repl(m: re.Match) -> str:
            inner = m.group("body")
            patched = _patch_people_block(inner)
            return "People,\n" + patched + "\n;"

        text = people_rx.sub(_people_repl, text)

        # 5) Write patched IDF
        out_path = pathlib.Path(self.out_dir) / f"{src.stem}__annual_co2_minimal_clean.idf"
        pathlib.Path(out_path).write_text(text.rstrip() + "\n")

        # 6) Wipe stale outputs if requested
        if wipe_outputs:
            for fn in ("eplusout.sql","eplusout.err","eplusout.audit","eplusout.eso","eplusout.mtr",
                    "eplusout.rdd","eplusout.mdd","eplusout.edd","audit.out","sqlite.err"):
                try: os.remove(os.path.join(self.out_dir, fn))
                except Exception: pass

        # 7) Activate
        if activate:
            if not getattr(self, "_orig_idf_path", None):
                self._orig_idf_path = self.idf
            self.idf = str(out_path)
            if reset:
                self.reset_state()

        self._co2_outdoor_schedule = schedule_name
        self._co2_per_zone_schedules = {}  # numeric approach
        return str(out_path)

    # ---------- catalog/RDD helpers ----------
    @staticmethod
    def _parse_sectioned_catalog(raw) -> Dict[str, List[List[str]]]:
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
        rows = list(_csv.reader(io.StringIO(text)))
        sections, cur = {}, None
        for r in rows:
            if not r: continue
            left = (r[0] or "").strip()
            right = r[1] if len(r) > 1 else ""
            if left.startswith("**") and left.endswith("**"):
                cur = left.strip("* ").upper()
                sections.setdefault(cur, [])
                continue
            if cur and right:
                try:
                    fields = ast.literal_eval(str(right))
                    if not isinstance(fields, (list, tuple)):
                        fields = [str(right)]
                except Exception:
                    fields = [str(right)]
                sections[cur].append(list(fields))
        return sections

    def _variables_from_catalog(self, state, *, discover_zone_keys=True,
                                discover_environment_keys=True, verify_handles=True) -> List[Dict]:
        secs = self._parse_sectioned_catalog(self.api.exchange.list_available_api_data_csv(state))
        base = []
        for f in secs.get("VARIABLES", []):
            name  = (f[0] if len(f) > 0 else "").strip()
            key   = (f[1] if len(f) > 1 else "").strip()
            units = (f[2] if len(f) > 2 else "").strip()
            desc  = (f[3] if len(f) > 3 else "").strip()
            if name:
                base.append({"name": name, "key": key, "units": units, "desc": desc, "handle": -1})

        if not base:
            return []

        out = []
        for row in base:
            h = -1
            if verify_handles and row["key"]:
                h = self.api.exchange.get_variable_handle(state, row["name"], row["key"])
            row["handle"] = h
            if (not verify_handles) or (h != -1) or (row["key"] == ""):
                row["source"] = "api"
                out.append(row)

        if discover_zone_keys:
            zones = self.api.exchange.get_object_names(state, "Zone") or []
            zoneish = [r for r in base if "zone" in r["name"].lower()]
            for r in zoneish:
                for z in zones:
                    h = self.api.exchange.get_variable_handle(state, r["name"], z)
                    if h != -1:
                        out.append({"name": r["name"], "key": z, "units": r["units"],
                                    "desc": r["desc"], "handle": h, "source": "api"})

        if discover_environment_keys:
            envish = [r for r in base if any(tok in r["name"].lower() for tok in ("site", "weather", "outdoor"))]
            for r in envish:
                h = self.api.exchange.get_variable_handle(state, r["name"], "Environment")
                if h != -1:
                    out.append({"name": r["name"], "key": "Environment", "units": r["units"],
                                "desc": r["desc"], "handle": h, "source": "api"})

        seen, dedup = set(), []
        for r in out:
            k = (r["name"], r["key"])
            if k in seen:
                continue
            seen.add(k)
            dedup.append(r)
        return dedup

    def _parse_units_label(self, s: str) -> tuple[str, str]:
        """Split 'Some Name [kWh]' -> ('Some Name', 'kWh'); tolerates missing units."""
        s = (s or "").strip()
        if not s:
            return "", ""
        lb = s.rfind('[')
        rb = s.rfind(']')
        if lb != -1 and rb != -1 and rb > lb:
            return s[:lb].strip(), s[lb+1:rb].strip()
        return s, ""

    def _variables_from_rdd(self, dir_path: str) -> list[dict]:
        """
        Parse variables from eplusout.rdd.

        Supports BOTH:
        A) Newer 3-column style (no frequency, no key):
        VarType, VarReportType, Variable Name [Units]
        e.g. "Zone,Average,Zone People Convective Heating Energy [J]"

        B) Legacy dictionary-like styles with frequency tokens:
        <Key>, <Name [Units]>, <Frequency>
        <ObjType>, <Key>, <Name [Units]>, <Frequency>

        Returns rows with fields:
        kind='var', name, key='', freq='', units, desc=VarReportType (if present), handle=-1, source='rdd'
        """
        import os, re, pathlib

        rdd_path = os.path.join(dir_path, "eplusout.rdd")
        if not os.path.exists(rdd_path):
            return []

        text = pathlib.Path(rdd_path).read_text(errors="ignore")

        # Normalize + filter lines
        lines = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('!'):
                continue
            if ln.endswith(';'):
                ln = ln[:-1].rstrip()
            # Skip header-ish lines
            if ln.startswith("Program Version") or "Var Type" in ln and "Variable Name" in ln:
                continue
            lines.append(ln)

        # Frequency tokens used in older formats; if present → legacy branch
        freq_tokens = {
            'DETAILED','TIMESTEP','ZONE TIMESTEP','ZONETIMESTEP','SYSTEM TIMESTEP',
            'SYSTEMTIMESTEP','HOURLY','DAILY','MONTHLY','RUNPERIOD','RUN PERIOD'
        }

        def _split_units(label: str) -> tuple[str, str]:
            s = (label or "").strip()
            lb = s.rfind('['); rb = s.rfind(']')
            if lb != -1 and rb != -1 and rb > lb:
                return s[:lb].strip(), s[lb+1:rb].strip()
            return s, ""

        rows, seen = [], set()

        for ln in lines:
            parts = [p.strip() for p in ln.split(',')]

            # --- Legacy: last token is a known frequency ---
            if len(parts) >= 3 and parts[-1].upper().replace('_', ' ') in freq_tokens:
                freq = parts[-1].strip()
                label = parts[-2].strip()
                # key is token before label; works for 3- or 4-field legacy lines
                key = parts[-3].strip() if len(parts) >= 3 else ""
                name, units = _split_units(label)
                if not name:
                    continue
                sig = ("legacy", name, key, freq)
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append({
                    "kind": "var",
                    "name": name,
                    "key": key,
                    "freq": freq,
                    "units": units,
                    "desc": "",          # legacy lines don’t carry VarReportType
                    "handle": -1,
                    "source": "rdd",
                })
                continue

            # --- New: 3-column "VarType, VarReportType, Name [Units]" ---
            if len(parts) >= 3:
                var_type = parts[0]            # e.g., "Zone", "Site", ...
                report_type = parts[1]         # "Average", "Sum", "Minimum", ...
                # Variable name may contain commas in rare cases → join the rest back
                label = ",".join(parts[2:]).strip()
                name, units = _split_units(label)
                if not name:
                    continue
                sig = ("new", name, report_type)
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append({
                    "kind": "var",
                    "name": name,
                    "key": "",                  # RDD new style does not list keys
                    "freq": "",                 # RDD does not encode reporting frequency
                    "units": units,
                    "desc": report_type,        # keep the Var Report Type for reference
                    "handle": -1,
                    "source": "rdd",
                })
                continue

            # Otherwise: ignore unrecognized line
            continue

        return rows

    def _meters_from_dictionary(self, dir_path: str) -> list[dict]:
        """
        Parse meters from eplusout.mdd if present; otherwise, parse category/stat style
        lines directly from eplusout.rdd (e.g., 'HVAC, Sum, Meter Name [hr]').
        """
        import os, re, pathlib
        rows, seen = [], set()

        # 1) Prefer .mdd if it exists
        mdd_path = os.path.join(dir_path, "eplusout.mdd")
        if os.path.exists(mdd_path):
            text = pathlib.Path(mdd_path).read_text(errors="ignore")
            # Pattern: "Category, <Stat>, <Name [Units]>"
            rx = re.compile(r'(?im)^\s*(?P<cat>[^,;\n]+)\s*,\s*(?P<stat>[^,;\n]+)\s*,\s*(?P<label>[^\n]+?)\s*$')
            for m in rx.finditer(text):
                cat = (m.group("cat") or "").strip()
                stat = (m.group("stat") or "").strip()
                label = (m.group("label") or "").strip()
                nm, units = self._parse_units_label(label)
                # Require units to reduce accidental captures
                if not nm or not units:
                    continue
                sig = (nm, units, cat, stat)
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append({
                    "kind": "meter",
                    "name": nm,
                    "key": cat,      # store category (HVAC, InteriorLights, etc.)
                    "freq": "",      # meters don't have a fixed freq in dict
                    "units": units,
                    "desc": stat,    # Sum, Average, etc.
                    "handle": -1,
                    "source": "mdd",
                })

        # 2) Also scan .rdd for the same pattern (some builds print meters there)
        rdd_path = os.path.join(dir_path, "eplusout.rdd")
        if os.path.exists(rdd_path):
            text = pathlib.Path(rdd_path).read_text(errors="ignore")
            rx = re.compile(r'(?im)^\s*(?P<cat>[^,;\n]+)\s*,\s*(?P<stat>[^,;\n]+)\s*,\s*(?P<label>[^\n]+?)\s*$')

            # 1) Tokens that clearly denote variable "VarType" rows → exclude as meters
            var_type_tokens = {
                "ZONE","SITE","SYSTEM","PLANT","NODE","SURFACE","SPACE","PEOPLE","ENVIRONMENT",
                "AIRLOOP","AIR LOOP","BRANCH","OUTDOORAIR","OUTDOOR AIR","COIL","FAN","PUMP"
            }
            # 2) Variable-name prefixes to exclude (avoid classifying them as meters)
            var_name_prefixes = (
                "Site ", "Zone ", "Space ", "People ", "System ", "Surface ", "Zone HVAC ", "System Node "
            )

            freq_tokens = {"DETAILED","TIMESTEP","ZONE TIMESTEP","SYSTEM TIMESTEP","HOURLY","DAILY","MONTHLY","RUNPERIOD","RUN PERIOD"}

            for m in rx.finditer(text):
                cat = (m.group("cat") or "").strip()
                stat = (m.group("stat") or "").strip()
                label = (m.group("label") or "").strip()

                # skip if this looks like a variable row (new 3-col RDD)
                if cat.upper() in var_type_tokens:
                    continue
                if any(label.startswith(pfx) for pfx in var_name_prefixes):
                    continue

                # Skip if the tail token looks like a variable frequency (legacy)
                tail = label.split(",")[-1].strip().upper()
                if tail in freq_tokens:
                    continue

                nm, units = self._parse_units_label(label)
                if not nm or not units:
                    continue
                sig = (nm, units, cat, stat)
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append({
                    "kind": "meter",
                    "name": nm,
                    "key": cat,
                    "freq": "",
                    "units": units,
                    "desc": stat,
                    "handle": -1,
                    "source": "rdd",
                })
        return rows

    def _collect_dictionary_from_dir(self, dir_path: str, want: set[str]) -> list[dict]:
        """Try to parse dictionary files already in dir_path. Returns requested kinds only."""
        out = []
        if "var" in want:
            out.extend(self._variables_from_rdd(dir_path))
        if "meter" in want:
            out.extend(self._meters_from_dictionary(dir_path))
        return out

    def _enrich_freq_from_sql_and_idf(self, rows: list[dict]) -> list[dict]:
        """
        Fill 'freq' for variables when RDD lacked it, using:
        - eplusout.sql ReportDataDictionary (preferred if present)
        - else Output:Variable blocks in the current IDF.
        Leaves existing non-empty 'freq' as-is.
        """
        import os, sqlite3, pathlib

        # Build (key,name) -> freq map from SQL if available (prefer the most-populated freq)
        freq_map: dict[tuple[str,str], tuple[str,int]] = {}
        sql_path = os.path.join(self.out_dir or "", "eplusout.sql")
        if os.path.exists(sql_path):
            conn = sqlite3.connect(sql_path)
            try:
                for kv, nm, fq, n in conn.execute("""
                    SELECT COALESCE(d.KeyValue,''), d.Name, COALESCE(d.ReportingFrequency,''), COUNT(*) AS n
                    FROM ReportData r
                    JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                    GROUP BY d.Name, d.KeyValue, d.ReportingFrequency
                """):
                    if not fq:
                        continue
                    k = (str(kv).strip(), str(nm).strip())
                    prev = freq_map.get(k)
                    if not prev or n > prev[1]:
                        freq_map[k] = (fq, n)
            finally:
                conn.close()

        # Fallback: scan Output:Variable in the active IDF
        if not freq_map:
            src = pathlib.Path(self.idf)
            text = src.read_text(errors="ignore")
            # you already have a scanner:
            for key, name, fq in self._scan_output_variables(text):  # returns (key,name,freq)
                freq_map[(key, name)] = (fq, 1)

        # Apply to rows that have blank freq
        def _apply_one(r: dict):
            if r.get("kind") != "var" or r.get("freq"):
                return
            nm = (r.get("name") or "").strip()
            ky = (r.get("key") or "").strip()
            # exact key
            m = freq_map.get((ky, nm))
            if not m and ky not in ("", "*"):
                # wildcard/blank variants
                m = freq_map.get(("*", nm)) or freq_map.get(("", nm))
            if m:
                r["freq"] = m[0]

        for r in rows:
            _apply_one(r)
        return rows

    def list_variables_safely(
        self,
        *,
        kinds=("var", "meter"),
        discover_zone_keys=True,
        discover_environment_keys=True,
        verify_handles=True,
        save_csv=True
    ) -> list[dict]:
        """
        Discover reportable **variables** and/or **meters** for the active model using a
        robust, fallback-heavy strategy that avoids fragile dependencies on any single
        artifact.

        Discovery order
        ---------------
        1) **Existing dictionary files in `out_dir`** (fast; comprehensive):
        - Parses `eplusout.rdd` for variables and (when present) meters,
        - Parses `eplusout.mdd` for meters.
        Returns rows with `source` set to `"rdd"` / `"mdd"`.
        2) **Generate dictionaries in a temporary directory** (no writes to `out_dir`):
        - If not already present, patches a temporary IDF with:
            `Output:VariableDictionary, IDF;`
        - Runs EnergyPlus once (design-day not required here) to emit RDD/MDD,
            then parses those.
        3) **Live API catalog fallback (variables only)**:
        - Runs a tiny **design-day** in a temp dir, then queries the runtime API
            catalog after warmup.
        - Can optionally auto-discover per-zone keys and environment keys and
            (optionally) verify handles.
        - Meters are **not** available from this path.

        Parameters
        ----------
        kinds : Sequence[str], default ("var", "meter")
            What to return: `"var"`, `"meter"`, or both. The alias `"both"` is also
            accepted. Order is ignored.
        discover_zone_keys : bool, default True
            (API fallback only) Expand zone-like variables by enumerating **Zone**
            object keys when possible.
        discover_environment_keys : bool, default True
            (API fallback only) Add environment-keyed variables (e.g., *Site…* with
            key `"Environment"`).
        verify_handles : bool, default True
            (API fallback only) Attempt to resolve a `handle` for each (name, key)
            via `get_variable_handle`; unresolved items may be dropped.
            Dictionary-derived rows have `handle=-1`.
        save_csv : bool, default True
            Write a CSV summary to `out_dir` when rows are found:
            - `"variables_only.csv"`   if only variables,
            - `"meters_only.csv"`      if only meters,
            - `"dictionary_combined.csv"` if RDD/MDD were used,
            - `"variables_with_desc.csv"` if API fallback was used.

        Returns
        -------
        list[dict]
            Each record contains:
            `{"kind": "var"|"meter", "name": str, "key": str, "units": str,
            "desc": str, "handle": int, "source": "rdd"|"mdd"|"api"}`

            Notes:
            - The legacy `'freq'` field (when present in some sources) is **removed**.
            - Dictionary-derived rows usually have `handle=-1`.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        (Otherwise, errors during probing are caught; when all fallbacks fail,
        the function returns an empty list after logging a warning.)

        Notes
        -----
        - The dictionary probe is preferred because it lists **all** variables/meters
        that would be reportable from the current IDF, regardless of whether they
        appear in Output objects yet.
        - The API fallback only includes variables that the runtime catalog exposes;
        it cannot list meters, and availability may differ by EnergyPlus version.
        - Temporary runs use a **fresh state** and a temp directory; they do not touch
        `out_dir` or the active state.

        Examples
        --------
        Get everything from whatever source is available (and save CSV):

        >>> rows = util.list_variables_safely()
        >>> rows[0]
        {'kind': 'var', 'name': 'Zone Air Temperature', 'key': 'SPACE1-1',
        'units': 'C', 'desc': 'Average', 'handle': -1, 'source': 'rdd'}

        Variables only, forcing API behavior (if no RDD/MDD exist), and ensuring
        zone/environment expansions with handle verification:

        >>> rows = util.list_variables_safely(
        ...     kinds=('var',),
        ...     discover_zone_keys=True,
        ...     discover_environment_keys=True,
        ...     verify_handles=True
        ... )

        Meters only (requires RDD/MDD path to succeed):

        >>> meters = util.list_variables_safely(kinds=('meter',))
        >>> [m['name'] for m in meters[:5]]
        ['Electricity:Facility', 'Cooling:Electricity', 'Heating:Electricity', ...]
        """
        import tempfile, re, os, shutil, subprocess, pathlib

        def _norm_kinds(k):
            if isinstance(k, str):
                k = (k,)
            s = {x.strip().lower() for x in k}
            if "both" in s:
                s = {"var", "meter"}
            return s & {"var", "meter"}

        want = _norm_kinds(kinds)
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."

        # 1) Use dictionaries already present in main out_dir
        direct = self._collect_dictionary_from_dir(self.out_dir, want)
        if direct:
            rows = direct
            used_rdd = True
        else:
            rows = []
            used_rdd = False
            rdd_error = None
            # 2) Temp-dir probe to generate dictionaries
            try:
                with tempfile.TemporaryDirectory() as tdir:
                    src_path = pathlib.Path(self.idf)
                    txt = src_path.read_text(errors="ignore")
                    if not re.search(r'^\s*Output\s*:\s*VariableDictionary\s*,', txt, flags=re.I | re.M):
                        txt = txt.rstrip() + "\n\nOutput:VariableDictionary,\n  IDF;\n"
                    tmp_idf = os.path.join(tdir, src_path.stem + "_with_rdd.idf")
                    pathlib.Path(tmp_idf).write_text(txt)
                    eplus_bin = shutil.which("energyplus") or "energyplus"
                    subprocess.run([eplus_bin, "-w", self.epw, "-d", tdir, tmp_idf], check=True)

                    rows = self._collect_dictionary_from_dir(tdir, want)
                    used_rdd = True
            except Exception as e:
                rdd_error = e
                rows = []

            # 3) Fallback to API catalog (variables only)
            if not rows and "var" in want:
                try:
                    state = self.api.state_manager.new_state()
                    bucket = {"vars": []}
                    def after_warmup(s):
                        vars_live = self._variables_from_catalog(
                            s,
                            discover_zone_keys=discover_zone_keys,
                            discover_environment_keys=discover_environment_keys,
                            verify_handles=verify_handles,
                        ) or []
                        for r in vars_live:
                            r["kind"] = "var"
                            # do NOT set any default freq
                            r.setdefault("source", "api")
                        bucket["vars"] = vars_live
                        try:
                            self.api.runtime.stop_simulation(s)
                        except Exception:
                            pass

                    with tempfile.TemporaryDirectory() as tdir2:
                        self.api.runtime.callback_after_new_environment_warmup_complete(state, after_warmup)
                        self.api.runtime.run_energyplus(state, ['-w', self.epw, '-d', tdir2, '--design-day', self.idf])
                        self.api.state_manager.reset_state(state)

                    rows = bucket["vars"]
                    used_rdd = False
                    if "meter" in want:
                        try:
                            self._log(1, "[list_variables_safely] Catalog fallback cannot provide meters; dictionary probe failed.")
                        except Exception:
                            print("[list_variables_safely] Catalog fallback cannot provide meters; dictionary probe failed.")
                except Exception as e2:
                    try:
                        self._log(1, f"[list_variables_safely] Failed. Dictionary error={rdd_error!r}, catalog error={e2!r}")
                    except Exception:
                        print(f"[list_variables_safely] Failed. Dictionary error={rdd_error!r}, catalog error={e2!r}")
                    return []

        # 🔻 Remove 'freq' key from all rows (if present from legacy paths)
        for r in rows:
            r.pop("freq", None)

        # Save CSV (no 'freq' column)
        if save_csv and rows:
            if want == {"var"}:
                fname = "variables_only.csv"
            elif want == {"meter"}:
                fname = "meters_only.csv"
            else:
                fname = "dictionary_combined.csv" if used_rdd else "variables_with_desc.csv"
            path = os.path.join(self.out_dir, fname)
            fieldnames = ["kind","name","key","units","desc","handle","source"]  # <-- no 'freq'
            with open(path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
            try:
                self._log(1, f"Saved dictionary → {path} (n={len(rows)})")
            except Exception:
                print(f"Saved dictionary → {path} (n={len(rows)})")

        return rows

    def _list_actuators_in_run(self, state, *, save_dir=None, verify_handles=True) -> List[Dict]:
        secs = self._parse_sectioned_catalog(self.api.exchange.list_available_api_data_csv(state))
        acts = []
        if "ACTUATORS" in secs and secs["ACTUATORS"]:
            for f in secs["ACTUATORS"]:
                comp = (f[0] if len(f)>0 else "").strip()
                ctrl = (f[1] if len(f)>1 else "").strip()
                key  = (f[2] if len(f)>2 else "").strip()
                unit = (f[3] if len(f)>3 else "").strip()
                acts.append({"component_type": comp, "control_type": ctrl, "actuator_key": key, "units": unit})
        # Also consider schedules
        for typ in self._SCHEDULE_TYPES:
            for name in (self.api.exchange.get_object_names(state, typ) or []):
                acts.append({"component_type": typ, "control_type": "Schedule Value", "actuator_key": name, "units": ""})
        # Dedup + optional verify
        seen, out = set(), []
        for a in acts:
            key3 = (a["component_type"], a["control_type"], a["actuator_key"])
            if key3 in seen: continue
            seen.add(key3)
            h = self.api.exchange.get_actuator_handle(state, *key3)
            a["handle"] = h
            if (not verify_handles) or (h != -1):
                out.append(a)
        if save_dir:
            path = os.path.join(save_dir, "actuators.csv")
            with open(path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["component_type","control_type","actuator_key","units","handle"])
                w.writeheader(); w.writerows(out)
            self._log(1, f"Saved actuators → {path} (n={len(out)})")
        return out

    def _list_controllables_api_only(
        self,
        *,
        verify_handles: bool = True,
        expand_people: bool = True,
        include_schedules: bool = True,
        save_csv: bool = True
    ) -> list[dict]:
        """
        Enumerate controllable variables (actuators) WITHOUT relying on eplusout.edd.
        Uses the runtime API catalog + object names.

        Returns rows with:
        component_type, control_type, actuator_key, units, handle, source

        Notes:
        - 'handle' values are resolved in a temp state and are NOT reusable across runs.
        - People expansion targets control types exactly as listed by the API for 'People'
        (commonly 'Number of People', sometimes 'Activity Level').
        """
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."

        import tempfile, pathlib, re, os

        # 1) First tiny run: pull catalog + live object names
        bucket = {"acts": [], "people_names": [], "sched_names": {}}

        def _after_warmup(s):
            # base list from API (no verification at this stage)
            base = self._list_actuators_in_run(s, save_dir=None, verify_handles=False)  # uses catalog + schedule scan
            bucket["acts"] = base or []
            try:
                bucket["people_names"] = self.api.exchange.get_object_names(s, "People") or []
            except Exception:
                bucket["people_names"] = []

            if include_schedules:
                scheds = {}
                for typ in self._SCHEDULE_TYPES:
                    try:
                        scheds[typ] = list(self.api.exchange.get_object_names(s, typ) or [])
                    except Exception:
                        scheds[typ] = []
                bucket["sched_names"] = scheds

            # stop asap
            try:
                self.api.runtime.stop_simulation(s)
            except Exception:
                pass

        state = self.api.state_manager.new_state()
        with tempfile.TemporaryDirectory() as tdir:
            self.api.runtime.callback_after_new_environment_warmup_complete(state, _after_warmup)
            self.api.runtime.run_energyplus(state, ['-w', self.epw, '-d', tdir, '--design-day', self.idf])
            self.api.state_manager.reset_state(state)

        base_rows = bucket["acts"][:]
        people_names = bucket["people_names"][:]

        # 2) Expand People actuators with wildcard/blank keys → concrete People objects
        expanded: list[dict] = []
        for a in base_rows:
            comp = (a.get("component_type") or "").strip()
            ctrl = (a.get("control_type") or "").strip()
            key  = (a.get("actuator_key") or "").strip()
            units = (a.get("units") or "").strip()
            src = a.get("source") or "api"

            if expand_people and comp.lower() == "people" and (key in ("", "*", "ALL")):
                # common controllables: "Number of People", sometimes "Activity Level"
                for pname in people_names:
                    expanded.append({
                        "component_type": comp,
                        "control_type": ctrl,
                        "actuator_key": pname,
                        "units": units,
                        "handle": -1,
                        "source": src + "-expanded"
                    })
            else:
                expanded.append({
                    "component_type": comp,
                    "control_type": ctrl,
                    "actuator_key": key,
                    "units": units,
                    "handle": -1,
                    "source": src
                })

        # 3) (Optional) add Schedule Value actuators explicitly (in case catalog omitted any)
        if include_schedules:
            for typ, names in (bucket.get("sched_names") or {}).items():
                for nm in names:
                    expanded.append({
                        "component_type": typ,
                        "control_type": "Schedule Value",
                        "actuator_key": nm,
                        "units": "",
                        "handle": -1,
                        "source": "api-schedule"
                    })

        # 4) Deduplicate triplets (component_type, control_type, actuator_key)
        seen, dedup = set(), []
        for r in expanded:
            t = (r["component_type"], r["control_type"], r["actuator_key"])
            if t in seen:
                continue
            seen.add(t); dedup.append(r)

        # 5) Verify handles in a fresh state (and optionally filter)
        if verify_handles:
            vstate = self.api.state_manager.new_state()
            def _after_input(vs):
                for r in dedup:
                    try:
                        h = self.api.exchange.get_actuator_handle(
                            vs, r["component_type"], r["control_type"], r["actuator_key"]
                        )
                    except Exception:
                        h = -1
                    r["handle"] = h
                try:
                    self.api.runtime.stop_simulation(vs)
                except Exception:
                    pass

            with tempfile.TemporaryDirectory() as tdir2:
                self.api.runtime.callback_after_component_get_input(vstate, _after_input)
                self.api.runtime.run_energyplus(vstate, ['-w', self.epw, '-d', tdir2, '--design-day', self.idf])
                self.api.state_manager.reset_state(vstate)

            dedup = [r for r in dedup if r.get("handle", -1) != -1]

        # 6) Save CSV
        if save_csv and dedup:
            path = os.path.join(self.out_dir, "controllables_api_only.csv")
            with open(path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["component_type","control_type","actuator_key","units","handle","source"])
                w.writeheader(); w.writerows(dedup)
            self._log(1, f"Saved controllables → {path} (n={len(dedup)})")

        return dedup

    def list_actuators_safely(self, *, verify_handles=True) -> List[Dict]:
        """
        Enumerate controllable **actuators** for the active model using a safe,
        API-only workflow (no dependency on `eplusout.edd`). This is a thin,
        backward-compatible wrapper around `_list_controllables_api_only(...)`.

        What it does
        ------------
        - Runs a tiny **design-day** in a temporary directory to query the runtime
        actuator catalog and live object names.
        - Expands wildcard/blank **People** actuators (e.g., `"Number of People"`)
        into concrete `People` object keys.
        - Includes **Schedule Value** actuators for common schedule types
        (`Schedule:Compact`, `Schedule:Constant`, `Schedule:File`, `Schedule:Year`).
        - (Optional) **verifies** handles in a fresh state so returned items are
        actually controllable for this IDF/EPW.
        - Writes a CSV summary to `<out_dir>/controllables_api_only.csv`.

        Parameters
        ----------
        verify_handles : bool, default True
            Resolve a `handle` for each `(component_type, control_type, actuator_key)`
            via `get_actuator_handle`. When `True`, rows without a valid handle are
            filtered out. (Note: these handles are resolved in a **temporary state**
            and are **not reusable** across runs.)

        Returns
        -------
        List[Dict]
            Each record has:
            `{"component_type": str,
            "control_type": str,
            "actuator_key": str,
            "units": str,
            "handle": int,        # -1 if not verified or unavailable
            "source": str}`       # e.g., "api", "api-expanded", "api-schedule"

        Requirements
        ------------
        Call `set_model(idf, epw, out_dir)` first; this method asserts that `idf`,
        `epw`, and `out_dir` are configured.

        Notes
        -----
        - This approach does **not** require `eplusout.edd` and is resilient across
        EnergyPlus versions.
        - CSV output is only written when at least one actuator is discovered.

        Examples
        --------
        Discover and verify all controllables; save CSV:

        >>> acts = util.list_actuators_safely()
        >>> acts[0]
        {'component_type': 'People', 'control_type': 'Number of People',
        'actuator_key': 'OPEN OFFICE PEOPLE', 'units': '', 'handle': 12345,
        'source': 'api-expanded'}

        Skip handle verification (keep all catalog entries):

        >>> acts = util.list_actuators_safely(verify_handles=False)
        """
        return self._list_controllables_api_only(
            verify_handles=verify_handles,
            expand_people=True,
            include_schedules=True,
            save_csv=True
        )

    def flatten_mtd(
        self,
        *,
        dir_path: str | None = None,
        generate_if_missing: bool = False,
        save_csv: bool = True,
    ) -> list[dict]:
        """
        Flatten eplusout.mtd into pairwise mappings (variable ↔ meter).

        Returns list of dict records with fields:
        - kind: "mtd_map"
        - direction: "var_on_meter" (from “Meters for … OnMeter = …”)
                        or "meter_contains" (from “For Meter = … contents are:”)
        - meter, meter_units
        - variable, variable_key, variable_units
        - var_index (int or None)  # from "Meters for <index>, ..."
        - resource_type (only when provided in the file)
        - source: "mtd"

        If generate_if_missing=True and eplusout.mtd is absent, runs a tiny
        design-day in a temp dir to try producing it.

        Notes on format: see EnergyPlus docs for eplusout.mtd examples.  [oai_citation:1‡Big Ladder Software](https://bigladdersoftware.com/epx/docs/8-7/output-details-and-examples/eplusout-mtd.html)
        """
        assert self.out_dir, "Call set_model(idf, epw, out_dir) first."
        import os, re, pathlib, tempfile, subprocess, shutil

        def _parse_units_label(s: str) -> tuple[str, str]:
            s = (s or "").strip()
            lb = s.rfind('['); rb = s.rfind(']')
            if lb != -1 and rb != -1 and rb > lb:
                return s[:lb].strip(), s[lb+1:rb].strip()
            return s, ""

        def _split_key_and_name(label: str) -> tuple[str, str]:
            # "SPACE1-1:Lights Electric Energy" -> ("SPACE1-1", "Lights Electric Energy")
            if ":" in label:
                left, right = label.split(":", 1)
                return left.strip(), right.strip()
            return "", label.strip()

        def _read_or_generate_mtd(path_out: str) -> tuple[str, bool]:
            """Return path to an mtd (either in out_dir or a temp dir). (path, is_temp)"""
            if os.path.exists(path_out):
                return path_out, False
            if not generate_if_missing:
                raise FileNotFoundError(path_out)
            # try to generate via a tiny design-day run
            assert self.idf and self.epw, "Need idf/epw to generate .mtd"
            with tempfile.TemporaryDirectory() as tdir:
                # Run EnergyPlus; .mtd is produced alongside other outputs
                eplus_bin = shutil.which("energyplus") or "energyplus"
                subprocess.run([eplus_bin, "-w", self.epw, "-d", tdir, self.idf], check=True)
                alt = os.path.join(tdir, "eplusout.mtd")
                if not os.path.exists(alt):
                    raise FileNotFoundError("eplusout.mtd was not produced during the temp run.")
                # Copy to out_dir (optional); here we parse directly from temp
                return alt, True

        outdir = dir_path or self.out_dir
        mtd_path = os.path.join(outdir, "eplusout.mtd")
        mtd_path, is_temp = _read_or_generate_mtd(mtd_path)
        text = pathlib.Path(mtd_path).read_text(errors="ignore")

        records: list[dict] = []
        seen_pairs: set[tuple[str, str, str, str]] = set()  # (direction, meter, vkey, vname)

        # --- Regexes for the two sections ---
        rx_meters_for = re.compile(
            r'(?im)^\s*Meters\s+for\s+(?P<idx>\d+)\s*,\s*(?P<label>.+?)\s*$'
        )
        rx_onmeter = re.compile(
            r'(?im)^\s*OnMeter\s*=\s*(?P<meter>[^\[\n]+?)\s*(?:\[(?P<munits>[^\]]+)\])?\s*$'
        )
        rx_meter_header = re.compile(
            r'(?im)^\s*For\s+Meter\s*=\s*(?P<meter>[^\[,]+?)\s*'
            r'(?:\[(?P<munits>[^\]]+)\])?\s*'
            r'(?:,\s*ResourceType\s*=\s*(?P<rtype>[^,]+?)\s*)?'
            r',?\s*contents\s+are\s*:\s*$'
        )

        lines = text.splitlines()
        i, n = 0, len(lines)

        while i < n:
            ln = lines[i].rstrip()
            m_head_var = rx_meters_for.match(ln)
            m_head_meter = rx_meter_header.match(ln)

            # Section 1: "Meters for <index>, <variable label [units]>"
            if m_head_var:
                idx = int(m_head_var.group("idx"))
                var_label_raw = m_head_var.group("label").strip()
                # The sample shows the variable label ending with [units] on the same line.
                var_label, v_units_line = _parse_units_label(var_label_raw)
                v_key, v_name = _split_key_and_name(var_label)

                # consume subsequent "OnMeter = ..." lines
                i += 1
                while i < n:
                    ln2 = lines[i].rstrip()
                    if not ln2 or rx_meters_for.match(ln2) or rx_meter_header.match(ln2):
                        # next section or blank → stop this block
                        break
                    m_on = rx_onmeter.match(ln2)
                    if m_on:
                        meter = (m_on.group("meter") or "").strip()
                        m_units = (m_on.group("munits") or "").strip()
                        sig = ("var_on_meter", meter, v_key.lower(), v_name.lower())
                        if sig not in seen_pairs:
                            seen_pairs.add(sig)
                            records.append({
                                "kind": "mtd_map",
                                "direction": "var_on_meter",
                                "meter": meter,
                                "meter_units": m_units,
                                "variable": v_name,
                                "variable_key": v_key,
                                "variable_units": v_units_line,
                                "var_index": idx,
                                "resource_type": "",
                                "source": "mtd",
                            })
                    i += 1
                continue

            # Section 2: "For Meter = <meter> [units], (optional ResourceType=...), contents are:"
            if m_head_meter:
                meter = (m_head_meter.group("meter") or "").strip()
                m_units = (m_head_meter.group("munits") or "").strip()
                rtype = (m_head_meter.group("rtype") or "").strip()

                i += 1
                while i < n:
                    ln2 = lines[i].rstrip()
                    if not ln2 or rx_meters_for.match(ln2) or rx_meter_header.match(ln2):
                        break
                    # component line; typically "<key>:<Variable Name>"
                    comp = ln2.strip().lstrip("-").strip()
                    if not comp:
                        i += 1
                        continue
                    v_key, v_name = _split_key_and_name(comp)
                    # units not shown on these lines; leave blank
                    sig = ("meter_contains", meter, v_key.lower(), v_name.lower())
                    if sig not in seen_pairs:
                        seen_pairs.add(sig)
                        records.append({
                            "kind": "mtd_map",
                            "direction": "meter_contains",
                            "meter": meter,
                            "meter_units": m_units,
                            "variable": v_name,
                            "variable_key": v_key,
                            "variable_units": "",
                            "var_index": None,
                            "resource_type": rtype,
                            "source": "mtd",
                        })
                    i += 1
                continue

            i += 1

        # Optional CSV
        if save_csv and records:
            path = os.path.join(self.out_dir, "meter_details_flat.csv")
            fieldnames = [
                "kind","direction","meter","meter_units",
                "variable","variable_key","variable_units",
                "var_index","resource_type","source"
            ]
            with open(path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in records:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
            self._log(1, f"[mtd] Wrote {path} (n={len(records)})")

        return records

    # ---------- occupancy (People actuators) ----------

    def _occ_default_map_zone_to_people(self, s, zones_subset=None, *, verbose=True) -> dict[str, list[str]]:
        """
        Heuristic: map each zone to People objects whose names contain the zone token.
        Only uses live names (after inputs parsed).
        """
        z2p: dict[str, list[str]] = {}
        try:
            people_names = self.api.exchange.get_object_names(s, "People") or []
        except Exception:
            people_names = []

        zset = set(zones_subset or [])
        for p in people_names:
            for z in zset:
                if z.replace(" ", "").lower() in p.replace(" ", "").lower():
                    z2p.setdefault(z, []).append(p)

        if verbose:
            missing = [z for z in zset if z not in z2p]
            for z in missing:
                self._log(1, f"[OCC] Note: no People object matched zone '{z}' (will be ignored).")
        return z2p

    def _occ_current_timestamp(self, s):
        """Build a pandas.Timestamp aligned with our CSV timeline (year 2002, hour at interval START)."""
        m = self.api.exchange.month(s)
        d = self.api.exchange.day_of_month(s)
        h = max(0, self.api.exchange.hour(s) - 1)
        N = max(1, self.api.exchange.num_time_steps_in_hour(s))
        ts = max(1, self.api.exchange.zone_time_step_number(s))
        minute = int(round((ts - 1) * 60 / N))
        return pd.Timestamp(year=2002, month=m, day=d, hour=h, minute=minute)

    def _occ_cb_tick(self, s):
        # only if feature enabled and data present
        if not getattr(self, "_occ_enabled", False) or self._occ_df is None:
            return

        # wait until warmup is done
        if self.api.exchange.warmup_flag(s):
            return

        occ_zone_cols = list(self._occ_df.columns)
        # lazy init once
        if not self._occ_ready:
            if not self._zone_to_people:
                self._zone_to_people = self._occ_default_map_zone_to_people(
                    s, occ_zone_cols, verbose=self._occ_verbose
                )

            self._people_handles = {}
            total = 0
            for z, ppl_list in (self._zone_to_people or {}).items():
                for p in (ppl_list or []):
                    try:
                        h = self.api.exchange.get_actuator_handle(s, "People", "Number of People", p)
                    except Exception:
                        h = -1
                    if h != -1:
                        self._people_handles.setdefault(z, []).append(h)
                        total += 1

            self._occ_ready = True
            if self._occ_verbose:
                if total == 0:
                    self._log(1, "[OCC] WARNING: No People actuator handles resolved. (No People objects or names mismatch?)")
                else:
                    zones_ok = sum(1 for z in self._people_handles if self._people_handles[z])
                    self._log(1, f"[OCC] Resolved {total} People handles across {zones_ok} zones.")

        # apply CSV occupancy each system timestep
        t = self._occ_current_timestamp(s)
        df = self._occ_df
        if t in df.index:
            row = df.loc[t]
        else:
            # honor runtime policy for gap filling
            tmp = df.reindex(df.index.union([t])).sort_index()
            if self._occ_fill == "interpolate":
                tmp = tmp.interpolate(method="time").ffill()
            else:
                tmp = tmp.ffill()
            row = tmp.loc[t]

        for z in occ_zone_cols:
            val = float(row.get(z, 0.0) or 0.0)
            handles = self._people_handles.get(z, [])
            if not handles:
                continue
            per = val / max(1, len(handles))
            for h in handles:
                self.api.exchange.set_actuator_value(s, h, per)

    # ---------- run methods (single, non-duplicated) ----------

    def run_annual(self) -> int:
        """
        Run a full **annual** EnergyPlus simulation with the currently configured
        `idf`, `epw`, and `out_dir`.

        What this method does
        ---------------------
        1) Verifies that `set_model(idf, epw, out_dir)` has been called and that
        `out_dir` is writable.
        2) Proactively removes common stale outputs
        (`eplusout.sql`, `eplusout.err`, `eplusout.audit`) to avoid locked/dirty
        files causing SQLite/open errors.
        3) Resets the EnergyPlus API **state** (clean run).
        4) Re-registers any queued callbacks (e.g., those added via
        `register_begin_iteration(...)`, `register_after_hvac_reporting(...)`,
        `enable_runtime_logging()`, or CSV occupancy handlers).
        5) Executes EnergyPlus with arguments:
        `['-w', self.epw, '-d', self.out_dir, self.idf]`.

        Returns
        -------
        int
            EnergyPlus process exit code (0 indicates success).

        Raises
        ------
        AssertionError
            If `idf`, `epw`, or `out_dir` is not set (call `set_model(...)` first).

        Side effects
        ------------
        - Writes standard EnergyPlus outputs into `out_dir` (e.g., `eplusout.err`,
        `eplusout.eso`, `eplusout.mtr`, etc.).
        - If your active IDF includes `Output:SQLite`, also writes `eplusout.sql`.

        Tips
        ----
        - If you need the SQLite database, ensure your active IDF includes
        `Output:SQLite` or call `ensure_output_sqlite(activate=True)` before running.
        - For a quick probe that only runs sizing/design days, use `run_design_day()`.

        Examples
        --------
        Basic annual run:
        >>> util.set_model("models/bldg.idf", "weather/site.epw", out_dir="runs/annual")
        >>> code = util.run_annual()
        >>> assert code == 0

        Ensure SQL is produced, then run:
        >>> util.ensure_output_sqlite()  # appends Output:SQLite and activates patched IDF
        >>> util.run_annual()
        """
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        self._assert_out_dir_writable()
        # prevent SQLite open errors from a stale file
        self.clear_eplus_outputs(("eplusout.sql", "eplusout.err", "eplusout.audit"))
        self.reset_state()
        self._register_callbacks()
        return self.api.runtime.run_energyplus(
            self.state, ['-w', self.epw, '-d', self.out_dir, self.idf]
        )

    def run_design_day(self) -> int:
        """
        Run a **design-day–only** EnergyPlus simulation for the active `idf`/`epw`
        into `out_dir`. This is a fast probe run (sizing periods only), useful for
        validating inputs, generating dictionaries, or triggering lightweight
        callbacks without simulating the full year.

        What this does
        --------------
        1) Verifies `set_model(idf, epw, out_dir)` has been called and that `out_dir`
        is writable.
        2) Removes common stale outputs (`eplusout.sql`, `eplusout.err`,
        `eplusout.audit`) to avoid file locks/SQLite open errors.
        3) Resets the EnergyPlus API **state** (clean run).
        4) Re-registers any queued callbacks (e.g., from
        `register_begin_iteration(...)`, `register_after_hvac_reporting(...)`,
        CSV occupancy, or `enable_runtime_logging()`).
        5) Executes EnergyPlus with:
        `['-w', self.epw, '-d', self.out_dir, '--design-day', self.idf]`.

        Returns
        -------
        int
            EnergyPlus process exit code (0 means success).

        Raises
        ------
        AssertionError
            If `idf`, `epw`, or `out_dir` is missing (call `set_model(...)` first).

        Notes
        -----
        - If your active IDF includes `Output:VariableDictionary, IDF;`, this run can
        emit `eplusout.rdd/.mdd` quickly.
        - If your active IDF includes `Output:SQLite`, `eplusout.sql` (for sizing
        periods) will be produced; downstream readers should set
        `include_design_days=True` if they want to query those data.
        - Handles registered during this method are attached to the fresh state used
        for this run.

        Examples
        --------
        Quick sizing-period check:
        >>> util.set_model("models/bldg.idf", "weather/site.epw", out_dir="runs/dd")
        >>> code = util.run_design_day()
        >>> assert code == 0

        Generate dictionaries fast, then parse:
        >>> util.ensure_output_sqlite()          # optional, if you also want eplusout.sql
        >>> util.run_design_day()
        >>> vars_and_meters = util.list_variables_safely()
        """
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        self._assert_out_dir_writable()
        self.clear_eplus_outputs(("eplusout.sql", "eplusout.err", "eplusout.audit"))
        self.reset_state()
        self._register_callbacks()
        return self.api.runtime.run_energyplus(
            self.state, ['-w', self.epw, '-d', self.out_dir, '--design-day', self.idf]
        )

    def dry_run_min(self, *, include_ems_edd: bool = False, reset: bool = True, design_day: bool = True) -> int:
        """
        Perform a **minimal probe run** that emits dictionary-style files to `out_dir`
        (fast, no parsing), then exits. Intended for quickly generating:
        - `eplusout.rdd` (report variables),
        - `eplusout.mdd` (report meters),
        - and optionally `eplusout.edd` (EMS diagnostics / actuator listings).

        What this does
        --------------
        1) Reads the active IDF and writes a lightweight, **temporary patched** IDF
        to `<out_dir>/<stem>__dicts.idf` that:
        - **Removes** any existing `Output:SQLite` block (prevents opening/writing
            `eplusout.sql` for this probe).
        - **Removes all EMS objects** (`EnergyManagementSystem:*`) from the temp file
            to avoid side effects during the dictionary run.
        - **Ensures** `Output:VariableDictionary, IDF;` so RDD/MDD are produced.
        - **Optionally** adds `Output:EnergyManagementSystem, Verbose, Verbose;`
            when `include_ems_edd=True` so `eplusout.edd` is generated.
        2) (Optional) Resets the EnergyPlus API state (`reset=True`) to guarantee a
        clean run.
        3) Runs EnergyPlus with:
        - `--design-day` when `design_day=True` (fast sizing-period probe),
        - otherwise a short annual pass using the patched IDF.
        4) Returns the raw EnergyPlus **exit code**. No parsing is performed here.

        Parameters
        ----------
        include_ems_edd : bool, default False
            If `True`, adds `Output:EnergyManagementSystem` to the patched IDF so
            `eplusout.edd` is emitted alongside RDD/MDD.
        reset : bool, default True
            If `True`, calls `reset_state()` before launching the probe run.
        design_day : bool, default True
            If `True`, runs with `--design-day` (much faster). If `False`, runs
            without it (annual-style invocation).

        Returns
        -------
        int
            EnergyPlus process exit code (0 indicates success).

        Raises
        ------
        AssertionError
            If `idf`, `epw`, or `out_dir` is not configured
            (call `set_model(idf, epw, out_dir)` first).

        Notes
        -----
        - This method **does not** register callbacks and **does not** touch
        `Output:SQLite`, so no `eplusout.sql` is created during this probe.
        - The patched IDF lives in `out_dir` and does not modify your original IDF.
        - Once files are emitted, you can use higher-level helpers (e.g.,
        `list_variables_safely()` which prefers in-place RDD/MDD in `out_dir`,
        or custom parsers) to read them.

        Examples
        --------
        Generate RDD/MDD quickly, then list variables/meters:

        >>> util.set_model("models/bldg.idf", "weather/site.epw", out_dir="runs/probe")
        >>> code = util.dry_run_min()          # emits eplusout.rdd/.mdd to runs/probe
        >>> assert code == 0
        >>> rows = util.list_variables_safely()  # now picks up RDD/MDD from out_dir

        Also emit EMS diagnostics (EDD):

        >>> util.dry_run_min(include_ems_edd=True)

        Run the probe without design-day (less common, slower):

        >>> util.dry_run_min(design_day=False)
        """
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        import pathlib, re

        # Patch a lightweight IDF in-place in out_dir
        src = pathlib.Path(self.idf)
        text = src.read_text(errors="ignore")

        
        # Strip Output:SQLite so this probe never opens eplusout.sql
        text = re.sub(
            r'(?is)^\s*Output\s*:\s*SQLite\s*,.*?;[ \t]*(?:\r?\n|$)',
            '',
            text,
            flags=re.MULTILINE
        )

        # Strip ALL EMS objects from the *temporary* file used for the dictionary run
        text = re.sub(
            r'(?is)^\s*EnergyManagementSystem\s*:[^;]+?;[ \t]*(?:\r?\n|$)',
            '',
            text,
            flags=re.MULTILINE
        )


        if not re.search(r'^\s*Output\s*:\s*VariableDictionary\s*,', text, flags=re.I | re.M):
            text = text.rstrip() + "\n\nOutput:VariableDictionary,\n  IDF;\n"

        if include_ems_edd and not re.search(r'^\s*Output\s*:\s*EnergyManagementSystem\s*,', text, flags=re.I | re.M):
            text = text.rstrip() + "\n\nOutput:EnergyManagementSystem,\n  Verbose,\n  Verbose;\n"

        patched = pathlib.Path(self.out_dir) / f"{src.stem}__dicts.idf"
        patched.write_text(text)

        if reset:
            self.reset_state()

        args = ['-w', self.epw, '-d', self.out_dir]
        if design_day:
            args.append('--design-day')
        args.append(str(patched))
        return self.api.runtime.run_energyplus(self.state, args)

    # ---------- set simulation parameters (patch IDF) ----------

    def clear_patched_idf(self):
        """Revert to the original IDF if we switched to a patched one."""
        if getattr(self, "_orig_idf_path", None):
            self.idf = self._orig_idf_path
        self._patched_idf_path = None
        self._orig_idf_path = None  # also clear to return to a clean slate

    def _remove_object_blocks(self, idf_text: str, obj_name: str) -> str:
        """Remove ALL blocks of a given object (case-insensitive; simple regex parser)."""
        pattern = rf'(?is)^\s*{re.escape(obj_name)}\s*,.*?;[ \t]*\n'
        return re.sub(pattern, '', idf_text, flags=re.MULTILINE)

    def _append_block(self, idf_text: str, block: str) -> str:
        return idf_text.rstrip() + "\n\n" + block.strip() + "\n"

    # ---- Output object scanning for strong dedupe ----
    @staticmethod
    def _scan_output_variables(text: str) -> set[Tuple[str,str,str]]:
        """
        Parse existing Output:Variable blocks into normalized (key, name, freq) tuples.
        Case-insensitive; trims quotes/whitespace.
        """
        pat = re.compile(r'(?is)^\s*Output:Variable\s*,(.*?);', re.MULTILINE)
        found = set()
        for blk in pat.findall(text):
            parts = [p.strip().strip(';').strip('"').strip("'") for p in blk.split(',')]
            if len(parts) >= 3:
                key, name, freq = parts[0], parts[1], parts[2]
                found.add((key.strip(), name.strip(), freq.strip()))
        return found

    @staticmethod
    def _scan_output_meters(text: str) -> set[Tuple[str]]:
        """
        Parse existing Output:Meter blocks into normalized (name, freq) tuples.
        """
        pat = re.compile(r'(?is)^\s*Output:Meter\s*,(.*?);', re.MULTILINE)
        found = set()
        for blk in pat.findall(text):
            parts = [p.strip().strip(';').strip('"').strip("'") for p in blk.split(',')]
            if len(parts) >= 2:
                name, freq = parts[0], parts[1]
                found.add((name.strip(), freq.strip()))
        return found

    def set_simulation_params(
        self,
        *,
        timestep_per_hour: Optional[int] = None,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None,
        start_day_of_week: Optional[str] = None,
        use_weather_holidays: bool = True,
        use_weather_dst: bool = True,
        weekend_holiday_rule: bool = False,
        use_weather_rain: bool = True,
        use_weather_snow: bool = True,
        min_warmup_days: Optional[int] = None,
        max_warmup_days: Optional[int] = None,
        runperiod_name: str = "RunPeriod 1",
        activate: bool = True,
        reset: bool = True
    ) -> str:
        """
        Patch the active IDF with common **simulation controls** (timestep, run period,
        warmup days), write it to `<out_dir>/<stem>__patched.idf`, and optionally
        activate it for subsequent runs.

        What gets modified
        ------------------
        - **Timestep**: replaces all existing `Timestep` objects with one line:
        `Timestep, <timestep_per_hour>;`
        - **RunPeriod**: replaces all existing `RunPeriod` objects with a single
        block named `runperiod_name`, spanning the requested `start` → `end`
        (month, day). `start_day_of_week` may be left blank to **use the weather
        file setting**.
        - **Sizing:Parameters**: replaces the object to set **Minimum/Maximum
        Number of Warmup Days** (other fields left blank -> defaults).

        Parameters
        ----------
        timestep_per_hour : int, optional
            Number of zone/system timesteps per hour (1–60). Example: `6` → 10-minute timestep.
        start, end : (int, int), optional
            Start/end as `(month, day)` (1-indexed). **Provide both** to set a RunPeriod,
            e.g. `start=(1, 1)`, `end=(12, 31)`.
        start_day_of_week : str, optional
            Day-of-week token (e.g., `"Monday"`, `"Tuesday"`, …). If `None` or empty,
            the field is left blank so EnergyPlus **uses the weather file**.
        use_weather_holidays, use_weather_dst : bool
            Whether to respect holidays and daylight saving from the EPW/holiday files.
        weekend_holiday_rule : bool
            Apply the weekend holiday rule (`Yes`/`No` in `RunPeriod`).
        use_weather_rain, use_weather_snow : bool
            Whether to use rain/snow indicators from the weather file.
        min_warmup_days, max_warmup_days : int, optional
            Positive integers controlling warmup bounds. Leave `None` to omit.
        runperiod_name : str
            Name for the generated `RunPeriod` object.
        activate : bool, default True
            If `True`, switch `self.idf` to the patched file and remember the original
            in `self._orig_idf_path`.
        reset : bool, default True
            If `activate=True`, also call `reset_state()` so the next run uses a clean state.

        Returns
        -------
        str
            Absolute path to the patched IDF written in `out_dir`.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` was not called.
        ValueError
            - If `timestep_per_hour` is outside `[1, 60]`.
            - If only one of `start`/`end` is provided.
            - If `min_warmup_days` or `max_warmup_days` is provided and < 1.

        Side effects
        ------------
        - Writes `<stem>__patched.idf` into `out_dir` and sets `self._patched_idf_path`.
        - If `activate=True`, sets `self.idf` to the patched path (non-destructive),
        and (when `reset=True`) resets the API state.
        - Previous `Timestep`, `RunPeriod`, and `Sizing:Parameters` blocks are **removed**
        from the patched file and replaced with your settings.

        Notes
        -----
        - This is a **text-level** patcher (robust for common cases) rather than a full IDF parser.
        - `RunPeriod` changes are irrelevant for `--design-day` runs; they apply to annual runs.
        - To revert to the original IDF later, call `clear_patched_idf()`.

        Examples
        --------
        Set a 10-minute timestep and run the full year:
        >>> util.set_simulation_params(timestep_per_hour=6)

        Simulate only Q1 using the weather file’s weekday alignment:
        >>> util.set_simulation_params(start=(1, 1), end=(3, 31))

        Tighten warmup bounds without changing run period:
        >>> util.set_simulation_params(min_warmup_days=6, max_warmup_days=30)

        Create a patched file **without** activating it (manual use later):
        >>> path = util.set_simulation_params(start=(7, 1), end=(9, 30), activate=False)
        >>> print(path)  # you can pass this path to EnergyPlus yourself

        Use a fixed start day (ignoring weather file alignment):
        >>> util.set_simulation_params(start=(1, 1), end=(12, 31), start_day_of_week="Monday")
        """
        ...
        assert self.idf and self.epw and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        src_path = pathlib.Path(self.idf)
        text = src_path.read_text(errors="ignore")
        if timestep_per_hour is not None:
            if not (1 <= int(timestep_per_hour) <= 60):
                raise ValueError("timestep_per_hour must be in [1, 60]")
            text = self._remove_object_blocks(text, "Timestep")
            text = self._append_block(text, f"Timestep,\n  {int(timestep_per_hour)};")
        if (start is not None) or (end is not None) or (start_day_of_week is not None):
            if not (start and end):
                raise ValueError("Provide both start=(month,day) and end=(month,day).")
            sm, sd = int(start[0]), int(start[1])
            em, ed = int(end[0]), int(end[1])
            text = self._remove_object_blocks(text, "RunPeriod")
            sdow = (start_day_of_week or "").strip()
            sdow_line = f"  {sdow}," if sdow else "  ,"
            rp_block = f"""RunPeriod,
            {runperiod_name},         !- Name
            {sm},                     !- Begin Month
            {sd},                     !- Begin Day of Month
            {em},                     !- End Month
            {ed},                     !- End Day of Month
            {sdow_line}                 !- Start Day of Week (blank = Use Weather File)
            {'Yes' if use_weather_holidays else 'No'},  !- Use Weather File Holidays and Special Days
            {'Yes' if use_weather_dst else 'No'},       !- Use Weather File Daylight Saving Period
            {'Yes' if weekend_holiday_rule else 'No'},  !- Apply Weekend Holiday Rule
            {'Yes' if use_weather_rain else 'No'},      !- Use Weather File Rain Indicators
            {'Yes' if use_weather_snow else 'No'};      !- Use Weather File Snow Indicators
            """
            text = self._append_block(text, rp_block)
        if (min_warmup_days is not None) or (max_warmup_days is not None):
            if (min_warmup_days is not None and min_warmup_days < 1) or (max_warmup_days is not None and max_warmup_days < 1):
                raise ValueError("warmup days must be positive integers")
            text = self._remove_object_blocks(text, "Sizing:Parameters")
            h = c = tconv = lconv = ""
            minw = str(int(min_warmup_days)) if min_warmup_days is not None else ""
            maxw = str(int(max_warmup_days)) if max_warmup_days is not None else ""
            sp_block = f"""Sizing:Parameters,
            {h},   !- Heating Sizing Factor
            {c},   !- Cooling Sizing Factor
            {tconv},   !- Timesteps in Averaging Window
            {lconv},   !- Heating Convergence Tolerance
            {minw},    !- Minimum Number of Warmup Days
            {maxw};    !- Maximum Number of Warmup Days
            """
            text = self._append_block(text, sp_block)
        out_path = pathlib.Path(self.out_dir) / f"{src_path.stem}__patched.idf"
        out_path.write_text(text)
        self._patched_idf_path = str(out_path)
        if activate:
            if not self._orig_idf_path:
                self._orig_idf_path = self.idf
            self.idf = self._patched_idf_path
            if reset:
                self.reset_state()
        return self._patched_idf_path

    # --- SQL-only helpers ---
    def ensure_output_sqlite(self, *, activate: bool = True, reset: bool = True) -> str:
        """
        Guarantee that the active IDF will produce **eplusout.sql** by ensuring an
        `Output:SQLite` object exists, then write a patched copy of the IDF to
        `<out_dir>/<stem>__sqlite.idf`. Optionally make this patched file the active
        `self.idf` and reset the runtime state so the next run uses it.

        Behavior
        --------
        - If the source IDF already contains an `Output:SQLite` object, it is **kept as-is**.
        - If not, a minimal block is appended:
            `Output:SQLite,`
            `  SimpleAndTabular;`
        - The result is saved as `__sqlite.idf` alongside other outputs.
        - When `activate=True`, the method:
            - Remembers the original path in `self._orig_idf_path` (once),
            - Points `self.idf` to the patched file,
            - Calls `reset_state()` if `reset=True`.

        Parameters
        ----------
        activate : bool, default True
            Switch `self.idf` to the patched `__sqlite.idf` immediately.
        reset : bool, default True
            If `activate=True`, reset the EnergyPlus state so subsequent runs use
            the newly patched IDF cleanly.

        Returns
        -------
        str
            Absolute path to the written `<stem>__sqlite.idf`.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called (requires `self.idf`
            and `self.out_dir`).

        Side Effects
        ------------
        - Writes a new IDF file in `out_dir`.
        - May update `self.idf` and reset the API state (depending on `activate` / `reset`).

        Notes
        -----
        - This method is **idempotent**: calling it again will just rewrite the same
        patched file (and keep any existing `Output:SQLite` blocks).
        - Producing `eplusout.sql` still requires running a simulation (e.g., via
        `run_design_day()` or `run_annual()`).
        - To add specific variables/meters to the SQL, pair with
        `ensure_output_variables(...)` / `ensure_output_meters(...)` before running.

        Examples
        --------
        Ensure SQL is enabled and run an annual simulation:
        >>> util.ensure_output_sqlite()
        >>> util.run_annual()
        >>> import os
        >>> os.path.exists(os.path.join(util.out_dir, "eplusout.sql"))
        True

        Prepare the patched file but keep the original IDF active:
        >>> path = util.ensure_output_sqlite(activate=False)
        >>> print(path)  # later you can switch util.idf to this path manually
        """
        assert self.idf and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        src = pathlib.Path(self.idf)
        text = src.read_text(errors="ignore")

        # Keep existing Output:SQLite if present; otherwise append a minimal one.
        if not re.search(r'^\s*Output:SQLite\s*,', text, flags=re.IGNORECASE | re.MULTILINE):
            text = text.rstrip() + "\n\nOutput:SQLite,\n  SimpleAndTabular;\n"

        out_path = pathlib.Path(self.out_dir) / f"{src.stem}__sqlite.idf"
        out_path.write_text(text)

        if activate:
            if not getattr(self, "_orig_idf_path", None):
                self._orig_idf_path = self.idf
            self.idf = str(out_path)
            if reset:
                self.reset_state()
        return str(out_path)

    def ensure_output_variables(self, specs: list[dict], *, activate: bool = True, reset: bool = True) -> str:
        """
        Ensure the IDF contains `Output:Variable` objects for the requested variables,
        writing a patched copy to `<out_dir>/<stem>__vars.idf`. Missing entries are
        appended; existing ones (by exact triplet **Key Value / Variable Name / Frequency**)
        are left untouched. Optionally switch `self.idf` to the patched file and reset
        the runtime state so the next run uses it.

        Parameters
        ----------
        specs : list of dict
            A list of variable specifications. Each dict supports:
            - **name** (str, required): EnergyPlus variable name
                (e.g., "Zone Air Temperature").
            - **key** (str, optional, default `"*"`): The Key Value (object name) to report.
                Use `"*"` to report for all keys that exist for that variable; use a specific
                zone/object name to target one key.
            - **freq** (str, optional, default `"TimeStep"`): Reporting frequency
                (e.g., "TimeStep", "Hourly", "Daily", "Monthly", "RunPeriod").
            Example item:
                `{"name": "Zone Air Temperature", "key": "*", "freq": "TimeStep"}`

        activate : bool, default True
            If True, switch `self.idf` to the newly written `__vars.idf`.
        reset : bool, default True
            If `activate=True`, call `reset_state()` so subsequent runs use the patched IDF.

        Returns
        -------
        str
            Absolute path to the written `<stem>__vars.idf`. If no new `Output:Variable`
            blocks were needed (all were already present), returns the original `self.idf`
            path without writing a new file.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        KeyError / ValueError
            If a spec is missing the required `"name"` field or contains invalid values.

        Behavior
        --------
        - Detects already-present `Output:Variable` entries via a strict triplet match:
        **(key, name, freq)** — comparisons are whitespace-trimmed.
        - Appends only the missing blocks and writes a patched copy to `out_dir`.
        - When activated for the first time, remembers the original path in
        `self._orig_idf_path` and updates `self.idf` to point at the patched file.

        Side Effects
        ------------
        - May write a new IDF (`__vars.idf`) to `out_dir`.
        - May update `self.idf` and reset the EnergyPlus state (depending on flags).

        Notes
        -----
        - This method only ensures variable **report requests** exist. To ensure the
        database file is produced, pair with `ensure_output_sqlite()` and then run
        a simulation (`run_design_day()` / `run_annual()`).
        - Use `ensure_output_meters(...)` to request meters instead of variables.

        Examples
        --------
        Request common zone temperatures for all zones at timesteps, and a few
        key-specific series hourly:

        >>> util.ensure_output_sqlite()  # ensure eplusout.sql will be produced
        >>> util.ensure_output_variables([
        ...     {"name": "Zone Air Temperature", "key": "*", "freq": "TimeStep"},
        ...     {"name": "Zone Mean Air Temperature", "key": "LIVING ZONE", "freq": "Hourly"},
        ...     {"name": "Zone Mean Air Temperature", "key": "KITCHEN", "freq": "Hourly"},
        ... ])
        '/abs/path/to/eplus_out/YourModel__vars.idf'

        Run a quick design-day simulation and then query data from SQL:

        >>> util.run_design_day()
        >>> df = util.get_sql_series_dataframe([
        ...     {"kind": "var", "name": "Zone Air Temperature", "key": "*", "label": "ZAT"}
        ... ])
        >>> df.head()
        """
        assert self.idf and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        src = pathlib.Path(self.idf)
        text = src.read_text(errors="ignore")

        existing = self._scan_output_variables(text)
        blocks = []
        for s in specs:
            nm = s["name"]
            ky = s.get("key", "*") or "*"
            fq = s.get("freq", "TimeStep")
            triplet = (ky.strip(), nm.strip(), fq.strip())
            if triplet in existing:
                continue
            block = f"""Output:Variable,
        {ky},                        !- Key Value
        {nm},                       !- Variable Name
        {fq};                       !- Reporting Frequency
        """
            blocks.append(block)

        if not blocks:
            return str(src)

        out_path = pathlib.Path(self.out_dir) / f"{src.stem}__vars.idf"
        out_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(b.rstrip() for b in blocks) + "\n")
        if activate:
            if not getattr(self, "_orig_idf_path", None):
                self._orig_idf_path = self.idf
            self.idf = str(out_path)
            if reset:
                self.reset_state()
        return str(out_path)

    def ensure_output_meters(self, names: list[str], *, freq: str = "TimeStep",
                            activate: bool = True, reset: bool = True) -> str:
        """
        Ensure the IDF contains `Output:Meter` objects for the requested meter names,
        writing a patched copy to `<out_dir>/<stem>__meters.idf`. Missing entries are
        appended; existing pairs **(Meter Name, Reporting Frequency)** are left as-is.
        Optionally switch `self.idf` to the patched file and reset the runtime state.

        Parameters
        ----------
        names : list[str]
            EnergyPlus meter names to request (e.g., `"Electricity:Facility"`,
            `"ElectricityPurchased:Facility"`, `"Gas:Facility"`, `"DistrictCooling:Facility"`).
        freq : str, default "TimeStep"
            Reporting frequency for all provided meters, commonly `"TimeStep"` or `"Hourly"`.
            (Other valid E+ frequencies like `"Daily"`, `"Monthly"`, `"RunPeriod"` will also work
            if you pass them explicitly.)
        activate : bool, default True
            If True, switch `self.idf` to the newly written `__meters.idf`.
        reset : bool, default True
            If `activate=True`, call `reset_state()` so subsequent runs use the patched IDF.

        Returns
        -------
        str
            Absolute path to the written `<stem>__meters.idf`. If all requested
            `(name, freq)` pairs already exist in the current IDF, returns the original
            `self.idf` path and does not write a new file.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.

        Behavior
        --------
        - Scans the active IDF for existing `Output:Meter` entries (normalized by
        `(name, freq)`), appends only missing ones, and writes a patched copy to `out_dir`.
        - When activated the first time, stores the original IDF path in `self._orig_idf_path`,
        updates `self.idf` to the patched file, and (optionally) resets the state.

        Notes
        -----
        - `Output:Meter` requests only control **what** is reported. To actually produce
        the SQLite database (`eplusout.sql`) that contains meter time series, ensure
        `Output:SQLite` is enabled (e.g., call `ensure_output_sqlite()`) and then run a
        simulation (`run_design_day()` / `run_annual()`).
        - For quick visualization, use `plot_sql_meters([...])`, which converts Joules to
        kWh by default when plotting.

        Examples
        --------
        Request whole-facility electricity and purchased electricity hourly, then run
        a design-day and plot:

        >>> util.ensure_output_sqlite()  # ensure eplusout.sql will be produced
        >>> util.ensure_output_meters(
        ...     ["Electricity:Facility", "ElectricityPurchased:Facility"],
        ...     freq="Hourly"
        ... )
        '/abs/path/to/eplus_out/YourModel__meters.idf'

        >>> util.run_design_day()
        >>> util.plot_sql_meters(
        ...     ["Electricity:Facility", "ElectricityPurchased:Facility"],
        ...     reporting_freq=("Hourly",),  # filter to hourly series
        ...     resample=None,               # no additional resampling
        ...     title="Facility Electricity (Hourly)"
        ... )
        """
        assert self.idf and self.out_dir, "Call set_model(idf, epw, out_dir) first."
        src = pathlib.Path(self.idf)
        text = src.read_text(errors="ignore")

        existing = self._scan_output_meters(text)
        blocks = []
        for nm in names:
            pair = (nm.strip(), freq.strip())
            if pair in existing:
                continue
            block = f"""Output:Meter,
            {nm},                       !- Meter Name
            {freq};                     !- Reporting Frequency
            """
            blocks.append(block)

        if not blocks:
            return str(src)

        out_path = pathlib.Path(self.out_dir) / f"{src.stem}__meters.idf"
        out_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(b.rstrip() for b in blocks) + "\n")
        if activate:
            if not getattr(self, "_orig_idf_path", None):
                self._orig_idf_path = self.idf
            self.idf = str(out_path)
            if reset:
                self.reset_state()
        return str(out_path)

    def inspect_sql_meter(self, name: str, *, include_design_days: bool = False):
        """
        Inspect a meter already written to `eplusout.sql` and summarize what’s available
        for it (reporting frequencies, units, and row counts).

        This helper queries the EnergyPlus SQLite output (`<out_dir>/eplusout.sql`)
        and returns a tidy `pandas.DataFrame` with one row per
        `(Name, ReportingFrequency, Units)` combination that exists in the database.
        By default it excludes sizing-period (design-day) environments so you see only
        regular simulation results.

        Parameters
        ----------
        name : str
            The meter name to inspect (e.g., `"Electricity:Facility"`,
            `"Gas:Facility"`, `"DistrictCooling:Facility"`).
        include_design_days : bool, default False
            If True, include rows from sizing/design-day environments
            (`EnvironmentName` like `'SizingPeriod:%'`). When False (default), those
            rows are filtered out.

        Returns
        -------
        pandas.DataFrame
            Columns:
            - **Name** (str) — meter name you requested.
            - **ReportingFrequency** (str) — e.g., `"TimeStep"`, `"Hourly"`,
            `"Daily"`, `"Monthly"`, `"RunPeriod"`.
            - **Units** (str) — units as recorded by EnergyPlus (often `"J"`).
            - **n_rows** (int) — number of time-series rows present for that
            `(Name, ReportingFrequency, Units)` combination.
            Sorted by `n_rows` descending.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` does not exist. (Enable SQLite output with
            `ensure_output_sqlite()` and run a simulation first.)

        Notes
        -----
        - This method does **not** modify your IDF or simulation settings; it only
        reads the SQLite database.
        - If you expect a frequency but don’t see it here, ensure the meter was
        requested at that frequency (e.g., via `ensure_output_meters(...)`) and
        re-run the simulation.

        Examples
        --------
        >>> util.ensure_output_sqlite()                     # make sure SQL will be produced
        >>> util.ensure_output_meters(["Electricity:Facility"], freq="Hourly")
        >>> util.run_design_day()                           # or util.run_annual()
        >>> util.inspect_sql_meter("Electricity:Facility")
            Name                   ReportingFrequency Units  n_rows
        0  Electricity:Facility               Hourly        J    8760
        1  Electricity:Facility               TimeStep      J   52560

        # Include sizing periods if you specifically want to see them:
        >>> util.inspect_sql_meter("Electricity:Facility", include_design_days=True)
        """

        assert self.out_dir, "Call set_model(...) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"{sql_path} not found.")
        conn = sqlite3.connect(sql_path)
        try:
            env_clause = "" if include_design_days else \
                "AND (ep.EnvironmentName IS NULL OR ep.EnvironmentName NOT LIKE 'SizingPeriod:%')"
            q = f"""
            SELECT d.Name, d.ReportingFrequency, COALESCE(d.Units,'') AS Units, COUNT(*) AS n_rows
            FROM ReportData r
            JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
            JOIN Time t ON r.TimeIndex = t.TimeIndex
            LEFT JOIN EnvironmentPeriods ep ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
            WHERE d.IsMeter = 1 AND d.Name = ?
                {env_clause}
            GROUP BY d.Name, d.ReportingFrequency, d.Units
            ORDER BY n_rows DESC
            """
            return pd.read_sql_query(q, conn, params=[name])
        finally:
            conn.close()

    def get_sql_series_dataframe(
        self,
        selections: list[dict],
        *,
        reporting_freq: tuple[str, ...] | None = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
    ) -> pd.DataFrame:
        """
        Query `<out_dir>/eplusout.sql` and return a tidy time-series DataFrame for the
        requested variables/meters.

        The result is **not** resampled or unit-converted; it is a clean, long-form
        table ready for plotting/aggregation elsewhere (e.g., `plot_sql_series`).

        Parameters
        ----------
        selections : list of dict
            One or more selection specs. Each spec supports:
            - **kind** : {"var","meter"} (default "var")
            - **name** : str — variable or meter name as recorded in SQL
            - **key**  : str | {"*", "ALL", ""} — for variables only
                * `""` or omitted → match blank/NULL key (and "Environment" for site vars)
                * `"*"` / `"ALL"` → expand to all keys that exist in SQL for `name`
                * specific key (e.g., `"SPACE1-1"`) → exact match (case-insensitive, trimmed)
            - **label** : str (optional) — human label for the series; if omitted the
            label defaults to `name` (and key/units may be appended for clarity).
        reporting_freq : tuple[str, ...] | None, default ("TimeStep", "Hourly")
            Reporting frequency filter applied in SQL. Use `None` to disable the
            frequency filter and return whatever exists (e.g., Hourly, Daily, etc).
            Matching is tolerant (e.g., "TIMESTEP" also matches "DETAILED").
        include_design_days : bool, default False
            If `True`, include rows from sizing/design-day environments. When `False`,
            they are filtered out (`EnvironmentName NOT LIKE 'SizingPeriod:%'`).

        Returns
        -------
        pandas.DataFrame
            Long-form dataframe with columns:
            - **timestamp** : pandas.Timestamp — interval **start** (E+ hours are
            end-of-interval; this shifts to the start). `Year==0` is mapped to 2002.
            - **trace** : str — display label assembled from `label`/`name` (+ key/units).
            - **value** : float — raw value from SQL (no resampling/unit conversion).
            - **kind** : {"var","meter"}
            - **name** : str — original variable/meter name
            - **key** : str — variable key (empty for meters)
            - **units** : str — units reported by EnergyPlus (e.g., `"J"`, `"C"`)
            - **freq** : str — reporting frequency recorded in SQL

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` does not exist (enable via
            `ensure_output_sqlite()` and rerun a simulation).
        ValueError
            If no rows match the request. The error message includes hints about
            available frequencies and a sample of existing keys.

        Notes
        -----
        - This function **does not** resample, window, or convert units. For example,
        meters remain in joules unless you convert later (e.g., to kWh).
        - When `key="*"`/`"ALL"`, the method fans out to **all** available keys for
        that variable name, returning a concatenated long table.
        - The `trace` column is constructed for direct plotting (unique legend labels).

        Examples
        --------
        Pull a single meter and a single zone variable (auto label and Hourly only):

        >>> df = util.get_sql_series_dataframe([
        ...     {"kind": "meter", "name": "Electricity:Facility"},
        ...     {"kind": "var", "name": "Zone Air Temperature", "key": "SPACE1-1", "label": "Space 1 Temp"},
        ... ], reporting_freq=("Hourly",))
        >>> df.head()
                timestamp             trace  value  kind  name                   key units   freq
        0  2002-01-01 00:00:00  Electricity...   ...  meter Electricity:Facility         J     Hourly
        1  2002-01-01 00:00:00      Space 1...   ...    var Zone Air Temperature  SPACE1-1  C  Hourly
        ...

        Expand a variable over **all** keys and disable frequency filtering:

        >>> df = util.get_sql_series_dataframe(
        ...     [{"kind":"var", "name":"Zone People Occupant Count", "key":"*"}],
        ...     reporting_freq=None
        ... )
        """

        assert self.out_dir, "Call set_model(idf, epw, out_dir) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"{sql_path} not found. Run ensure_output_sqlite() and re-run the sim.")

        conn = sqlite3.connect(sql_path)
        try:
            frames = []
            minute_col = self._sql_minute_col(conn)
            env_clause = "" if include_design_days else \
                "AND (ep.EnvironmentName IS NULL OR ep.EnvironmentName NOT LIKE 'SizingPeriod:%')"
            freq_clause, freq_params = self._sql_freq_clause(reporting_freq)

            for sel in selections:
                kind  = (sel.get("kind") or "var").lower()
                name  = sel.get("name")
                key   = sel.get("key")
                label = sel.get("label")

                if kind == "meter":
                    params = [name, *freq_params]
                    q = f"""
                    SELECT
                        d.Name AS name,
                        COALESCE(d.Units,'') AS units,
                        d.ReportingFrequency AS freq,
                        t.Year AS y, t.Month AS m, t.Day AS d, t.Hour AS h, t.{minute_col} AS mi,
                        r.Value AS val
                    FROM ReportData r
                    JOIN ReportDataDictionary d
                            ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                    JOIN Time t ON r.TimeIndex = t.TimeIndex
                    LEFT JOIN EnvironmentPeriods ep
                            ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                    WHERE d.IsMeter = 1
                        AND d.Name = ?
                        {freq_clause}
                        {env_clause}
                    """
                    rows = conn.execute(q, params).fetchall()
                    if not rows:
                        continue
                    df = pd.DataFrame(rows, columns=["name","units","freq","y","m","d","h","min","value"])
                    df["kind"] = "meter"
                    # trace = label or name [+ units]
                    if len(df) and df["units"].iloc[0]:
                        df["trace"] = (label or name) + f" [{df['units'].iloc[0]}]"
                    else:
                        df["trace"] = label or name
                    df["key"] = ""   # meters don’t carry KeyValue here
                    frames.append(df)

                else:
                    # Expand keys when asked
                    if key in ("*", "ALL"):
                        krows = conn.execute("""
                            SELECT DISTINCT COALESCE(KeyValue,'') AS kv
                            FROM ReportDataDictionary
                            WHERE (IsMeter = 0 OR IsMeter IS NULL) AND Name = ?
                        """, (name,)).fetchall()
                        keys = [kr[0] for kr in krows] or [""]
                    elif key in (None, ""):
                        keys = [""]
                    else:
                        keys = [key]

                    for k in keys:
                        if k:
                            key_clause = "AND (UPPER(TRIM(d.KeyValue)) = UPPER(TRIM(?)))"
                            params = [name, k, *freq_params]
                        else:
                            key_clause = "AND (d.KeyValue = '' OR d.KeyValue IS NULL)"
                            params = [name, *freq_params]

                        q = f"""
                        SELECT
                            d.Name AS name, COALESCE(d.KeyValue,'') AS vkey,
                            COALESCE(d.Units,'') AS units,
                            d.ReportingFrequency AS freq,
                            t.Year AS y, t.Month AS m, t.Day AS d, t.Hour AS h, t.{minute_col} AS mi,
                            r.Value AS val
                        FROM ReportData r
                        JOIN ReportDataDictionary d
                                ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                        JOIN Time t ON r.TimeIndex = t.TimeIndex
                        LEFT JOIN EnvironmentPeriods ep
                                ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                        WHERE (d.IsMeter = 0 OR d.IsMeter IS NULL)
                            AND d.Name = ?
                            {key_clause}
                            {freq_clause}
                            {env_clause}
                        """
                        rows = conn.execute(q, params).fetchall()
                        if not rows:
                            continue
                        df = pd.DataFrame(rows, columns=["name","vkey","units","freq","y","m","d","h","min","value"])
                        df["kind"] = "var"
                        base = label or name
                        # trace = base (+ key) [+ units]
                        df["trace"] = df.apply(lambda r: base + (f" ({r['vkey']})" if r.get("vkey") else ""), axis=1)
                        if len(df) and df["units"].iloc[0]:
                            df["trace"] = df["trace"] + " [" + df["units"] + "]"
                        df = df.rename(columns={"vkey": "key"})
                        frames.append(df)

            if not frames:
                # Build “what exists” tips to help the caller
                tips = []
                for sel in selections:
                    nm = sel.get("name")
                    rows = conn.execute(f"""
                        SELECT d.ReportingFrequency, COALESCE(d.Units,''), COUNT(*) AS n
                        FROM ReportData r
                        JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                        JOIN Time t ON r.TimeIndex = t.TimeIndex
                        LEFT JOIN EnvironmentPeriods ep ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                        WHERE d.Name = ? {env_clause}
                        GROUP BY d.ReportingFrequency, d.Units
                        ORDER BY n DESC
                    """, [nm]).fetchall()
                    if rows:
                        freqs = ", ".join(sorted({r[0] for r in rows if r[0]}))
                        keyrows = conn.execute("""
                            SELECT DISTINCT COALESCE(KeyValue,'') AS kv
                            FROM ReportDataDictionary
                            WHERE (IsMeter = 0 OR IsMeter IS NULL) AND Name = ?
                            ORDER BY kv
                        """, [nm]).fetchall()
                        keys_hint = ", ".join([str(k[0]) for k in keyrows][:8])
                        tips.append(f"- '{nm}' exists with frequencies: {freqs}" + (f" | keys (sample): {keys_hint}" if keys_hint else ""))

                extra = ("\n".join(tips) + "\n") if tips else ""
                raise ValueError(
                    "No rows matched. Check names/keys/frequencies, or set include_design_days=True.\n"
                    + extra + "Tip: reporting_freq=None disables the frequency filter."
                )

            df = pd.concat(frames, ignore_index=True)

            # timestamps (E+ hour is end-of-interval → shift to start; Year=0 → 2002)
            y = df["y"].replace(0, 2002)
            df["timestamp"] = pd.to_datetime(
                dict(year=y, month=df["m"], day=df["d"], hour=(df["h"] - 1).clip(lower=0), minute=df["min"]),
                errors="coerce"
            )
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            # Keep only the tidy columns the plotter needs
            return df[["timestamp","trace","value","kind","name","key","units","freq"]]
        finally:
            conn.close()

    def plot_sql_series(
        self,
        selections: list[dict],
        *,
        reporting_freq: tuple[str, ...] | None = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
        start=None,
        end=None,
        resample: str | None = "1H",
        meters_to_kwh: bool = True,
        aggregate_vars: str = "mean",   # when resampling variables
        title: str | None = None,
        show: bool = True,
        warn_rows_threshold: int = 50000,
    ):

    def plot_sql_series(
        self,
        selections: list[dict],
        *,
        reporting_freq: tuple[str, ...] | None = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
        start=None,
        end=None,
        resample: str | None = "1H",
        meters_to_kwh: bool = True,
        aggregate_vars: str = "mean",
        title: str | None = None,
        show: bool = True,
        warn_rows_threshold: int = 50000,
    ):
        """
        Plot one or more EnergyPlus variables/meters pulled directly from
        `<out_dir>/eplusout.sql`. This is a convenience wrapper around
        `get_sql_series_dataframe(...)` that optionally windows the data,
        converts meter energy from joules → kWh, resamples/aggregates, and
        renders an interactive Plotly line chart.

        Parameters
        ----------
        selections : list of dict
            Selection specs passed through to `get_sql_series_dataframe`. Each item:
            - **kind** : {"var","meter"} (default "var")
            - **name** : str — SQL variable/meter name
            - **key**  : str | {"*", "ALL", ""} — variables only
                * `""` / omitted → blank/NULL key (and "Environment" for site vars)
                * `"*"` / `"ALL"` → expand to all keys found in SQL
                * specific key (e.g., `"SPACE1-1"`) → exact, case-insensitive
            - **label** : str (optional) — friendly label used for the legend/trace
        reporting_freq : tuple[str, ...] | None, default ("TimeStep","Hourly")
            Frequency filter applied in SQL. Use `None` to disable filtering and
            accept any frequency present (Daily/Monthly/RunPeriod/etc).
            Matching is tolerant (e.g., "TIMESTEP" also matches "DETAILED").
        include_design_days : bool, default False
            Include sizing period rows (by default they are excluded).
        start, end : datetime-like or str, optional
            Inclusive window applied **after** SQL retrieval. Anything accepted by
            `pandas.to_datetime` (e.g., `"2002-01-03"`, `"2002-01-03 06:00"`).
        resample : str | None, default "1H"
            Pandas offset alias for time aggregation (e.g., `"15min"`, `"1H"`, `"1D"`).
            - If provided, data are resampled:
            * **meters** are **summed** over each bin
            * **variables** use `aggregate_vars` (e.g., `"mean"`, `"median"`, `"max"`)
            - If `None`, the original timestep data are plotted (can be very large).
        meters_to_kwh : bool, default True
            For meter series only, convert J → kWh (divide by 3.6e6), update the
            `units` to `"kWh"`, and adjust trace labels to end with `[kWh]`.
        aggregate_vars : str, default "mean"
            Aggregator name used for **variables** during resampling
            (anything supported by `Series.resample(...).agg(aggregate_vars)`).
            Ignored for meters (meters always sum).
        title : str | None, default None
            Figure title. If plotting a single series and `title` is `None`, the
            lone trace label is used as the title.
        show : bool, default True
            If `True`, calls `fig.show()`; otherwise only returns the figure.
        warn_rows_threshold : int, default 50000
            When `resample=None` and the plotted row count exceeds this value, a
            one-line warning is logged to help avoid sluggish rendering.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line chart with:
            - x-axis: **Time**
            - y-axis: **kWh** if `meters_to_kwh=True` and any meters are present,
            otherwise **Value**
            - multiple traces colored by the `trace` label when >1 series exist

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` is missing (enable via `ensure_output_sqlite()`
            and rerun a simulation).
        ValueError
            - If all rows were filtered out by `start`/`end`
            - If no data remain after resampling/aggregation
            - If the underlying `get_sql_series_dataframe` found no matches; the
            propagated error includes hints for available keys/frequencies.

        Notes
        -----
        - Time stamps are interval **starts** (EnergyPlus hours are end-of-interval).
        - For mixed inputs (meters + variables), meters are summed while variables
        use `aggregate_vars` during resampling.
        - If you need the underlying tidy dataframe without plotting, use
        `get_sql_series_dataframe(...)` directly.

        Examples
        --------
        Plot hourly facility electricity (kWh) and a single zone temperature:

        >>> util.ensure_output_sqlite()
        >>> fig = util.plot_sql_series(
        ...     selections=[
        ...         {"kind":"meter", "name":"Electricity:Facility"},
        ...         {"kind":"var",   "name":"Zone Air Temperature", "key":"SPACE1-1", "label":"Space 1 Tdb"}
        ...     ],
        ...     reporting_freq=("Hourly",),
        ...     start="2002-01-05", end="2002-01-12",
        ... )
        >>> fig  # interactive figure

        Plot all zone occupant counts at 15-minute resolution without converting meters:

        >>> fig = util.plot_sql_series(
        ...     selections=[{"kind":"var", "name":"Zone People Occupant Count", "key":"*"}],
        ...     reporting_freq=None,
        ...     resample="15min",
        ...     meters_to_kwh=False,
        ...     title="Occupants per Zone"
        ... )

        Render raw timestep data (no resample). A warning logs if very large:

        >>> fig = util.plot_sql_series(
        ...     selections=[{"kind":"meter","name":"Gas:Facility"}],
        ...     resample=None
        ... )
        """

        # ---- 1) fetch tidy data (no resample/window/convert) ----
        df = self.get_sql_series_dataframe(
            selections,
            reporting_freq=reporting_freq,
            include_design_days=include_design_days,
        )

        # ---- 2) optional window ----
        if start is not None:
            df = df[df["timestamp"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.to_datetime(end)]
        if df.empty:
            raise ValueError("All rows were filtered out by the time window; relax start/end or include_design_days=True.")

        # ---- 3) optional units conversion (meters: J → kWh) ----
        if meters_to_kwh:
            is_meter = df["kind"] == "meter"
            if is_meter.any():
                df.loc[is_meter, "value"] = df.loc[is_meter, "value"] / 3.6e6
                # update units + trace suffix to [kWh]
                df.loc[is_meter, "units"] = "kWh"
                df.loc[is_meter, "trace"] = df.loc[is_meter, "trace"].str.replace(
                    r"\s*\[.*\]$", " [kWh]", regex=True
                )

        # ---- 4) resample (sum meters, aggregate vars) ----
        if resample:
            pieces = []
            for tr, g in df.groupby("trace"):
                agg = "sum" if (g["kind"]=="meter").all() else aggregate_vars
                s = g.set_index("timestamp")["value"].resample(resample).agg(agg)
                pieces.append(s.rename(tr))
            wide = pd.concat(pieces, axis=1).reset_index()
            long = wide.melt(id_vars="timestamp", var_name="trace", value_name="value").dropna(subset=["value"])
        else:
            long = df[["timestamp","trace","value"]]
            if len(long) > warn_rows_threshold:
                try:
                    self._log(1, f"[plot_sql_series] Warning: rendering {len(long)} points (resample=None). Consider resampling.")
                except Exception:
                    print(f"[plot_sql_series] Warning: rendering {len(long)} points (resample=None). Consider resampling.")

        if long.empty:
            raise ValueError("No data to plot after resample/aggregation. Try a shorter resample or different frequency.")

        # ---- 5) plot ----
        if long["trace"].nunique() > 1:
            fig = px.line(long, x="timestamp", y="value", color="trace", title=title or "EnergyPlus Series (SQL)")
        else:
            ttl = title or long["trace"].unique()[0]
            fig = px.line(long, x="timestamp", y="value", title=ttl)

        ylab = "kWh" if meters_to_kwh and (df["kind"]=="meter").any() else "Value"
        fig.update_layout(xaxis_title="Time", yaxis_title=ylab)
        if show:
            fig.show()
        return fig

    def list_sql_zone_variables(
        self,
        *,
        name: str | None = None,            # exact variable name (e.g., "Zone Air Temperature")
        like: str = "Zone %",               # fallback: SQL LIKE pattern if 'name' is None
        reporting_freq: tuple[str, ...] | None = ("TimeStep","Hourly"),
        include_design_days: bool = False,
    ):
        """
        List zone-scoped variables that actually have rows in `<out_dir>/eplusout.sql`.

        This queries EnergyPlus's SQLite output (tables `ReportData*`, `Time`,
        and `EnvironmentPeriods`) and returns a summary table grouped by
        variable name, key (zone), units, and reporting frequency—along with a
        row count for each group.

        Parameters
        ----------
        name : str | None, default None
            Exact ReportDataDictionary `Name` to match (e.g., "Zone Air Temperature").
            When provided, `like` is ignored. Match is case-sensitive because it
            uses SQL equality against EnergyPlus's canonical casing.
        like : str, default "Zone %"
            SQL `LIKE` pattern used when `name` is None. Useful for discovery
            (e.g., `"Zone %Temperature%"`, `"Zone %CO2%"`).
        reporting_freq : tuple[str, ...] | None, default ("TimeStep","Hourly")
            Frequency filter applied in SQL. Use `None` to disable frequency
            filtering and include Daily/Monthly/RunPeriod rows if present.
            Matching is tolerant (e.g., "TIMESTEP" also matches "DETAILED").
        include_design_days : bool, default False
            If `True`, sizing period rows (`EnvironmentName` like "SizingPeriod:%")
            are included; otherwise they are excluded.

        Returns
        -------
        pandas.DataFrame
            Columns:
            - **Name** (str) — variable name
            - **KeyValue** (str) — zone (or blank for keyless variables)
            - **Units** (str)
            - **ReportingFrequency** (str)
            - **n_rows** (int) — number of rows matched in SQL

            Sorted by Name, KeyValue.

        Raises
        ------
        AssertionError
            If `set_model(idf, epw, out_dir)` has not been called.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` does not exist. (Call
            `ensure_output_sqlite()` and re-run a simulation.)

        Notes
        -----
        - This only reports variables that **actually appear in SQL** (i.e., you
        requested them via `Output:Variable` and ran the simulation).
        - Zone keys are returned exactly as stored in SQL (case preserved).
        - For quick plotting of a single zone variable across a few keys, see
        `plot_sql_zone_variable(...)`.

        Examples
        --------
        Discover all zone variables that contain "Temperature" (any frequency):

        >>> df = util.list_sql_zone_variables(like="Zone %Temperature%", reporting_freq=None)
        >>> df.head()

        Inspect which zones have data for Zone Air Temperature at TimeStep/Hourly:

        >>> df = util.list_sql_zone_variables(name="Zone Air Temperature",
        ...                                   reporting_freq=("TimeStep","Hourly"))
        >>> df.sort_values("n_rows", ascending=False).head()

        Include sizing periods for CO₂ variables:

        >>> util.list_sql_zone_variables(like="Zone %CO2%", include_design_days=True)
        """
        assert self.out_dir, "Call set_model(...) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"{sql_path} not found. Run ensure_output_sqlite() and a simulation first.")

        conn = sqlite3.connect(sql_path)
        try:
            env_clause = "" if include_design_days else \
                "AND (ep.EnvironmentName IS NULL OR ep.EnvironmentName NOT LIKE 'SizingPeriod:%')"

            if name is not None:
                name_clause = "AND d.Name = ?"
                params = [name]
            else:
                name_clause = "AND d.Name LIKE ?"
                params = [like]

            freq_clause, freq_params = self._sql_freq_clause(reporting_freq)
            params = params + freq_params

            q = f"""
                SELECT
                d.Name,
                d.KeyValue,
                d.Units,
                d.ReportingFrequency,
                COUNT(*) AS n_rows
                FROM ReportData r
                JOIN ReportDataDictionary d
                    ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                JOIN Time t ON r.TimeIndex = t.TimeIndex
                LEFT JOIN EnvironmentPeriods ep
                    ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                WHERE (d.IsMeter = 0 OR d.IsMeter IS NULL)
                {name_clause}
                {freq_clause}
                {env_clause}
                GROUP BY d.Name, d.KeyValue, d.Units, d.ReportingFrequency
                ORDER BY d.Name, d.KeyValue
            """
            return pd.read_sql_query(q, conn, params=params)
        finally:
            conn.close()

    def plot_sql_zone_variable(
        self,
        var_name: str,                       # e.g., "Zone Air Temperature"
        keys: list[str] | None = None,       # e.g., ["LIVING ZONE","KITCHEN"]; None → auto-pick top few
        *,
        reporting_freq: tuple[str, ...] | None = ("TimeStep","Hourly"),
        include_design_days: bool = False,
        start=None,
        end=None,
        resample: str | None = "1H",         # average variables hourly by default
        aggregate_vars: str = "mean",
        max_auto_keys: int = 4,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Plot a single zone variable for one or more zones (keys) from `eplusout.sql`.

        This is a convenience wrapper around `plot_sql_series(...)`. When `keys` is
        omitted, the method will discover zone keys that actually have rows for the
        requested variable (via `list_sql_zone_variables(...)`) and auto-select up to
        `max_auto_keys` with the most data.

        Parameters
        ----------
        var_name : str
            Exact variable name as stored in ReportDataDictionary (e.g., "Zone Air Temperature",
            "Zone Mean Air Temperature", "Zone Air CO2 Concentration").
        keys : list[str] | None, default None
            Zone names to plot. If `None`, the method inspects SQL and auto-picks up to
            `max_auto_keys` keys with the highest row counts for `var_name`.
        reporting_freq : tuple[str, ...] | None, default ("TimeStep","Hourly")
            Frequency filter. Use `None` to include any frequency present (Daily, Monthly, etc.).
        include_design_days : bool, default False
            Include sizing periods (`EnvironmentName` LIKE "SizingPeriod:%") when querying SQL.
        start, end : any, optional
            Optional time window. Parsed by `pandas.to_datetime`. Rows outside the window
            are filtered out before plotting.
        resample : str | None, default "1H"
            Pandas offset string to resample each trace before plotting (e.g., "30min",
            "2H"). Variables are aggregated using `aggregate_vars`. Use `None` to plot
            the native SQL time resolution.
        aggregate_vars : str, default "mean"
            Aggregation to apply when resampling variable traces (e.g., "mean", "median", "max").
        max_auto_keys : int, default 4
            Maximum number of keys to auto-select when `keys=None`.
        title : str | None, default None
            Plot title. If omitted, a title is synthesized as:
            `"{var_name} — key1, key2, key3…"`.
        show : bool, default True
            If True, calls `fig.show()`; the figure is always returned.

        Returns
        -------
        plotly.graph_objs.Figure
            A line chart with one trace per selected key.

        Raises
        ------
        ValueError
            - If no rows exist for `var_name` in SQL (when auto-picking keys).
            - If all rows are filtered out by the time window / frequency choices downstream.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` is missing. (Call `ensure_output_sqlite()` and rerun a simulation.)

        Notes
        -----
        - Units come from SQL and are appended to each trace label.
        - Variables are **not** converted (unlike meters which may be converted to kWh elsewhere).
        - This method delegates data retrieval + plotting to `get_sql_series_dataframe(...)`
        and `plot_sql_series(...)`.

        Examples
        --------
        Plot the top 4 zones for Zone Air Temperature, hourly-averaged:

        >>> util.plot_sql_zone_variable("Zone Air Temperature")

        Plot specific zones at 30-minute resolution:

        >>> util.plot_sql_zone_variable(
        ...     "Zone Air Temperature",
        ...     keys=["LIVING ZONE", "KITCHEN"],
        ...     reporting_freq=("TimeStep",),
        ...     resample="30min",
        ...     title="Living vs Kitchen — 30-min"
        ... )

        Focus on a summer window and include design-day outputs:

        >>> util.plot_sql_zone_variable(
        ...     "Zone Mean Air Temperature",
        ...     start="2002-06-01",
        ...     end="2002-08-31",
        ...     include_design_days=True
        ... )
        """
        if keys is None:
            df = self.list_sql_zone_variables(
                name=var_name, reporting_freq=reporting_freq, include_design_days=include_design_days
            )
            if df.empty:
                raise ValueError(
                    f"No rows found for variable '{var_name}'. "
                    f"Add Output:Variable objects or relax frequency/design-day filters."
                )
            df = df.sort_values(["n_rows"], ascending=False)
            keys = [k for k in df["KeyValue"].dropna().astype(str).tolist() if k][:max_auto_keys]

        sels = [{"kind":"var", "name": var_name, "key": k, "label": k} for k in keys]
        return self.plot_sql_series(
            selections=sels,
            reporting_freq=reporting_freq,
            include_design_days=include_design_days,
            start=start, end=end,
            resample=resample,
            meters_to_kwh=False,            # variables ≠ energy
            aggregate_vars=aggregate_vars,
            title=title or f"{var_name} — {', '.join(keys[:3])}{'…' if len(keys)>3 else ''}",
            show=show
        )

    def plot_sql_meters(self, meter_names: list[str], **kwargs):
        """
        Plot one or more EnergyPlus meters from `eplusout.sql`.

        This is a thin convenience wrapper around `plot_sql_series(...)` that builds
        the required selections for meters and forwards all keyword arguments.

        Parameters
        ----------
        meter_names : list[str]
            Exact meter names as stored in ReportDataDictionary (e.g.,
            "Electricity:Facility", "ElectricityPurchased:Facility",
            "NaturalGas:Facility"). Names must already exist in the SQL output.
        **kwargs :
            Any keyword arguments accepted by `plot_sql_series(...)`, including:
            - reporting_freq: tuple[str, ...] | None, default ("TimeStep","Hourly")
            - include_design_days: bool, default False
            - start, end: optional time window (parsed by pandas)
            - resample: str | None, default "1H"
            - meters_to_kwh: bool, default True (converts Joules → kWh for meters)
            - title: str | None
            - show: bool, default True

        Returns
        -------
        plotly.graph_objs.Figure
            A line chart with one trace per requested meter. If `meters_to_kwh=True`,
            the y-axis is in kWh; otherwise native meter units (typically Joules).

        Raises
        ------
        FileNotFoundError
            If `<out_dir>/eplusout.sql` does not exist (call `ensure_output_sqlite()`
            and rerun a simulation).
        ValueError
            If no rows are found after applying frequency / time filters.

        Notes
        -----
        - By default meters are converted from Joules to kWh for readability.
        Disable with `meters_to_kwh=False` to keep native units.
        - Use `inspect_sql_meter(name)` beforehand to discover available
        frequencies and units for a given meter.

        Examples
        --------
        Plot facility electricity (and purchased electricity if present), hourly:

        >>> util.plot_sql_meters(
        ...     ["Electricity:Facility", "ElectricityPurchased:Facility"],
        ...     reporting_freq=("TimeStep","Hourly"),
        ...     resample="1H",
        ...     title="Site Electricity — Hourly"
        ... )

        Plot natural gas without unit conversion, aggregated daily:

        >>> util.plot_sql_meters(
        ...     ["NaturalGas:Facility"],
        ...     meters_to_kwh=False,
        ...     resample="1D",
        ...     title="Daily Natural Gas (J)"
        ... )

        Focus on a specific window and include sizing periods:

        >>> util.plot_sql_meters(
        ...     ["Electricity:Facility"],
        ...     start="2002-07-01", end="2002-07-31",
        ...     include_design_days=True,
        ...     title="July Electricity — Including Sizing"
        ... )
        """
        sels = [{"kind":"meter", "name": m} for m in meter_names]
        return self.plot_sql_series(sels, **kwargs)

    def plot_sql_cov_heatmap(
        self,
        control_sels: list[dict],
        output_sels: list[dict],
        *,
        reporting_freq: tuple[str, ...] | None = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
        start=None,
        end=None,
        resample: str | None = "1h",
        reduce: str = "mean",
        meters_to_kwh: bool = True,
        stat: str = "cov",               # 'cov' or 'corr'
        min_periods: int = 8,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Compute and plot a **covariance / correlation heatmap** between “controls”
        and “outputs” extracted from `eplusout.sql`.

        Each entry in `control_sels` and `output_sels` is a selection dict compatible
        with `plot_sql_series(...)`, e.g.:

            {"kind": "var"|"meter", "name": "<E+ name>", "key": "<KeyValue|*|''>", "label": "<optional display name>"}

        The function:
        1) Pulls each selection via `plot_sql_series(..., show=False)` (so it reuses
            the same SQL pipeline, frequency filtering, time-windowing, resampling,
            and meter J→kWh conversion).
        2) Converts the returned traces into time-indexed `pandas.Series`.
        3) If a selection expands to multiple traces (e.g., multiple zone keys), it
            reduces them to a single series using `reduce`:
                - `"mean"` (default): average across traces by timestamp
                - `"sum"`: sum across traces
                - `"first"`: take the first trace as-is
        4) Concatenates all series, aligns them on timestamps, and computes the pairwise
            **covariance** (`stat="cov"`) or **correlation** (`stat="corr"`), using `min_periods`
            for robustness to missing data.
        5) Renders a heatmap with **Outputs** on the Y axis and **Controls** on the X axis.

        Parameters
        ----------
        control_sels : list[dict]
            One or more control-side selections (e.g., setpoints, HVAC power meters).
            Same schema as `plot_sql_series`. Each selection becomes one column in the heatmap
            (after optional reduction if the selection yields multiple traces).
        output_sels : list[dict]
            One or more output-side selections (e.g., zone temperatures, coil loads).
            Each selection becomes one row in the heatmap (post-reduction).
        reporting_freq : tuple[str, ...] | None, default ("TimeStep", "Hourly")
            SQL frequency filter passed through to `plot_sql_series`. Use `None` to disable.
        include_design_days : bool, default False
            If `False`, sizing periods are excluded. If a selection yields no data, the
            function **auto-retries once** with `include_design_days=True` for that selection.
        start, end : any, optional
            Optional time window. Passed through to `plot_sql_series` and applied before stats.
        resample : str | None, default "1h"
            Resampling step (Pandas offset alias) applied by `plot_sql_series`. Use `None`
            to keep native resolution. For variables, `aggregate_vars="mean"` is used;
            meters are summed by interval.
        reduce : {"mean","sum","first"}, default "mean"
            How to collapse multi-trace selections into a single series.
        meters_to_kwh : bool, default True
            If selections include meters, convert Joules → kWh (and relabel units) before stats.
        stat : {"cov","corr"}, default "cov"
            Matrix to compute/display: covariance or Pearson correlation.
        min_periods : int, default 8
            Minimum overlapping timestamps required for each pairwise stat (`pandas` behavior).
        title : str | None, optional
            Plot title. Defaults to "Covariance heatmap: outputs vs controls" (or "Correlation ...").
        show : bool, default True
            If True, displays the figure. The figure is returned either way.

        Returns
        -------
        plotly.graph_objects.Figure
            Heatmap with controls on X, outputs on Y. Color scale is zero-centered and
            symmetric (RdBu); colorbar label reflects the chosen statistic.

        Raises
        ------
        ValueError
            If no data are found for a required selection (even after the auto-retry),
            if there is no overlapping data to compute the matrix, or if the matrix is all-NaN.

        Notes
        -----
        - Series names are taken from `selection["label"]` when provided; otherwise from the
        E+ name (and key). If collisions occur, unique suffixes like " #2" are added.
        - This function **does not** alter units beyond the optional meter kWh conversion.
        - For large native time steps, consider resampling (e.g., `"15T"` or `"1H"`) to
        stabilize covariance/correlation and speed up plotting.

        Examples
        --------
        1) Correlate zone air temps (outputs) against thermostat setpoints (controls):

        >>> controls = [
        ...     {"kind": "var", "name": "Zone Thermostat Heating Setpoint Temperature", "key": "LIVING",  "label": "Heat SP — LIVING"},
        ...     {"kind": "var", "name": "Zone Thermostat Heating Setpoint Temperature", "key": "BEDROOM", "label": "Heat SP — BEDROOM"},
        ... ]
        >>> outputs = [
        ...     {"kind": "var", "name": "Zone Air Temperature", "key": "LIVING",  "label": "Tair — LIVING"},
        ...     {"kind": "var", "name": "Zone Air Temperature", "key": "BEDROOM", "label": "Tair — BEDROOM"},
        ... ]
        >>> fig = util.plot_sql_cov_heatmap(
        ...     control_sels=controls,
        ...     output_sels=outputs,
        ...     reporting_freq=("TimeStep","Hourly"),
        ...     resample="1H",
        ...     stat="corr",
        ... )

        2) Covariance between HVAC/equipment electricity (controls) and total cooling rates (outputs):

        >>> controls = [
        ...     {"kind": "meter", "name": "Electricity:HVAC", "label": "HVAC kWh"},
        ...     {"kind": "meter", "name": "Electricity:Facility", "label": "Facility kWh"},
        ... ]
        >>> outputs = [
        ...     {"kind": "var", "name": "Zone Total Cooling Rate", "key": "*", "label": "Zone cooling (avg)"},
        ... ]
        >>> fig = util.plot_sql_cov_heatmap(
        ...     control_sels=controls,
        ...     output_sels=outputs,
        ...     resample="1H",
        ...     reduce="mean",     # average across zones expanded by key="*"
        ...     stat="cov",
        ... )
        """

        import pandas as pd
        import numpy as np
        import plotly.express as px

        if not control_sels or not output_sels:
            raise ValueError("Provide at least one control selection and one output selection.")

        def _series_from_selection(sel: dict, *, _include_dd: bool):
            # pull via existing plot_sql_series (no SQL duplication)
            def _pull(_include_dd_flag: bool):
                return self.plot_sql_series(
                    selections=[sel],
                    reporting_freq=reporting_freq,
                    include_design_days=_include_dd_flag,
                    start=start,
                    end=end,
                    resample=resample,
                    meters_to_kwh=meters_to_kwh,
                    aggregate_vars="mean",
                    title=None,
                    show=False,
                )
            try:
                fig = _pull(_include_dd)
                if not fig.data:
                    raise ValueError("no-data")
            except Exception as e:
                # auto-retry including sizing periods if first pass yielded nothing
                if (not _include_dd) and ("No rows matched" in str(e) or "include_design_days=True" in str(e)):
                    fig = _pull(True)
                    if not fig.data:
                        raise
                else:
                    raise

            cols = []
            for tr in fig.data:
                nm = tr.name or sel.get("label") or sel.get("name")
                s = pd.Series(tr.y, index=pd.to_datetime(tr.x), name=str(nm))
                cols.append(s)
            dfw = pd.concat(cols, axis=1).sort_index()

            if dfw.shape[1] == 1 or reduce == "first":
                series = dfw.iloc[:, 0]
            elif reduce == "sum":
                series = dfw.sum(axis=1, min_count=1)
            else:
                series = dfw.mean(axis=1)

            return series.dropna().rename(str(sel.get("label") or sel.get("name") or "series"))

        # build labeled series
        controls, outputs, used = [], [], set()
        def _unique(lab: str):
            base = str(lab)
            if base not in used:
                used.add(base); return base
            i = 2
            while f"{base} #{i}" in used:
                i += 1
            used.add(f"{base} #{i}")
            return f"{base} #{i}"

        for cs in control_sels:
            s = _series_from_selection(cs, _include_dd=include_design_days)
            s.name = _unique(s.name); controls.append(s)
        for osel in output_sels:
            s = _series_from_selection(osel, _include_dd=include_design_days)
            s.name = _unique(s.name); outputs.append(s)

        # assemble matrix input
        df_all = pd.concat(controls + outputs, axis=1).sort_index()
        # (optional speed-up) downcast
        try: df_all = df_all.astype("float32", copy=False)
        except Exception: pass

        if stat.lower() == "corr":
            mat = df_all.corr(min_periods=min_periods); zlabel = "Correlation"
        else:
            mat = df_all.cov(min_periods=min_periods);   zlabel = "Covariance"

        if mat.empty:
            raise ValueError("Could not compute matrix (no overlapping data).")

        out_names = [s.name for s in outputs]
        ctl_names = [s.name for s in controls]
        sub = mat.loc[out_names, ctl_names]
        if sub.isna().all().all():
            raise ValueError("All entries are NaN. Not enough overlapping data across pairs.")

        vals = sub.values.astype(float)
        finite = np.isfinite(vals)
        vmax = float(np.nanmax(np.abs(vals[finite]))) if finite.any() else 1.0

        # Use range_color to zero-center; fallback to go.Heatmap if this PX version lacks range_color
        try:
            fig = px.imshow(
                vals,
                x=ctl_names,
                y=out_names,
                origin="lower",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                range_color=(-vmax, vmax),
                labels=dict(x="Controls", y="Outputs", color=zlabel),
                title=title or f"{zlabel} heatmap: outputs vs controls",
            )
        except TypeError:
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Heatmap(
                z=vals, x=ctl_names, y=out_names,
                colorscale="RdBu", zmin=-vmax, zmax=vmax,
                colorbar=dict(title=zlabel),
            ))
            fig.update_layout(
                title=title or f"{zlabel} heatmap: outputs vs controls",
                xaxis_title="Controls", yaxis_title="Outputs",
            )

        if show:
            fig.show()
        return fig

    # --------------- weather data ---------

    def export_weather_sql_to_csv(
        self,
        *,
        variables: list[str] | None = None,
        reporting_freq: tuple[str, ...] | None = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
        resample: str | None = "1H",      # average to hourly by default
        aggregate: str = "mean",          # 'mean', 'median', etc. for resample
        csv_filename: str = "weather_timeseries.csv",
        print_summary: bool = True,
    ):
        """
        Export weather (“Site …”) variables from `eplusout.sql` to a **wide** CSV and
        return a quick numeric summary.

        This helper pulls keyless/environment-scoped site variables from the EnergyPlus
        SQL output, aligns them on timestamps, optionally resamples/aggregates, writes a
        single wide table to `<out_dir>/<csv_filename>`, and returns both the path and a
        compact summary DataFrame.

        Parameters
        ----------
        variables : list[str] | None, default None
            Variable names to extract (e.g., "Site Outdoor Air Drybulb Temperature").
            If `None`, a useful default set is used:
            - "Site Outdoor Air Drybulb Temperature"
            - "Site Outdoor Air Dewpoint Temperature"
            - "Site Outdoor Air Humidity Ratio"
            - "Site Outdoor Air Barometric Pressure"
            - "Site Wind Speed"
            - "Site Wind Direction"
            - "Site Diffuse Solar Radiation Rate per Area"
            - "Site Direct Solar Radiation Rate per Area"
            - "Site Horizontal Infrared Radiation Rate per Area"
            - "Site Sky Temperature"

            Only entries with `KeyValue` of `''`, `NULL`, or `'Environment'` are selected.
            (Zone-scoped keys are intentionally ignored for weather export.)

        reporting_freq : tuple[str, ...] | None, default ("TimeStep", "Hourly")
            Filter by SQL reporting frequency. Pass `None` to disable frequency filtering.

        include_design_days : bool, default False
            Include sizing/design-day environments from the SQL (when False, they are excluded).

        resample : str | None, default "1H"
            Pandas offset alias for resampling the variables after alignment (e.g., "30T", "1D").
            Use `None` to keep native simulation resolution.

        aggregate : str, default "mean"
            Aggregation to apply per column during resampling (e.g., "mean", "median", "max").

        csv_filename : str, default "weather_timeseries.csv"
            Output CSV file name (written to `self.out_dir`). Columns:
            - "timestamp"
            - one column per variable, labeled as `"Name [units]"` when units are known.

        print_summary : bool, default True
            If True, logs a short summary and attempts to display the summary DataFrame.

        Returns
        -------
        (str, pandas.DataFrame)
            A tuple `(csv_path, summary_df)` where:
            - `csv_path` is the absolute path to the written CSV.
            - `summary_df` has columns: `["series", "rows", "min", "mean", "max"]`
            and `attrs`:
            - `window_start`: timestamp of first row in the CSV
            - `window_end`: timestamp of last row in the CSV
            - `rows`: number of rows in the CSV

        Raises
        ------
        AssertionError
            If `set_model(...)` has not been called to define `idf`, `epw`, and `out_dir`.
        FileNotFoundError
            If `<out_dir>/eplusout.sql` is missing. Call `ensure_output_sqlite()` and rerun the simulation.
        ValueError
            If none of the requested variables are found in SQL (a sample of available
            "Site %" names is included in the error message).

        Notes
        -----
        - Timestamps: EnergyPlus records times at **end-of-interval**; this shifts to the
        interval start for plotting/analysis consistency. `Year=0` is mapped to `2002`.
        - Units are **not** converted; each column header appends `[units]` when available.
        - Alignment: variables are outer-joined on timestamp; output is sorted by time.
        - This function only queries variables (not meters).

        Examples
        --------
        Export the default set at hourly resolution:

        >>> csv_path, summary = util.export_weather_sql_to_csv()

        Export a custom subset at native resolution (no resample) and include sizing periods:

        >>> csv_path, summary = util.export_weather_sql_to_csv(
        ...     variables=["Site Wind Speed", "Site Outdoor Air Drybulb Temperature"],
        ...     resample=None,
        ...     include_design_days=True,
        ...     reporting_freq=None
        ... )

        30-minute median aggregation and a custom file name:

        >>> csv_path, summary = util.export_weather_sql_to_csv(
        ...     resample="30T",
        ...     aggregate="median",
        ...     csv_filename="weather_30min_median.csv",
        ...     print_summary=False
        ... )

        Inspect the overall time window encoded in the summary:

        >>> summary.attrs["window_start"], summary.attrs["window_end"], summary.attrs["rows"]
        """


        assert self.out_dir, "Call set_model(idf, epw, out_dir) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(
                f"{sql_path} not found. Run `ensure_output_sqlite()` and re-run the simulation."
            )

        # default set of useful weather variables
        default_vars = [
            "Site Outdoor Air Drybulb Temperature",
            "Site Outdoor Air Dewpoint Temperature",
            "Site Outdoor Air Humidity Ratio",
            "Site Outdoor Air Barometric Pressure",
            "Site Wind Speed",
            "Site Wind Direction",
            "Site Diffuse Solar Radiation Rate per Area",
            "Site Direct Solar Radiation Rate per Area",
            "Site Horizontal Infrared Radiation Rate per Area",
            "Site Sky Temperature",
        ]
        variables = variables or default_vars

        conn = sqlite3.connect(sql_path)
        try:
            minute_col = self._sql_minute_col(conn)

            env_clause = "" if include_design_days else \
                "AND (ep.EnvironmentName IS NULL OR ep.EnvironmentName NOT LIKE 'SizingPeriod:%')"
            fclause, fparams = self._sql_freq_clause(reporting_freq)

            # pull one variable (keyless or 'Environment') into a tidy df
            def _pull_var(vname: str) -> pd.DataFrame:
                # accept KeyValue '' or NULL or 'Environment'
                params = [vname, *fparams]
                q = f"""
                SELECT
                    d.Name AS name,
                    COALESCE(d.KeyValue,'') AS vkey,
                    COALESCE(d.Units,'') AS units,
                    d.ReportingFrequency AS freq,
                    t.Year AS y, t.Month AS m, t.Day AS d, t.Hour AS h, t.{minute_col} AS mi,
                    r.Value AS val
                FROM ReportData r
                JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                JOIN Time t ON r.TimeIndex = t.TimeIndex
                LEFT JOIN EnvironmentPeriods ep ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                WHERE (d.IsMeter = 0 OR d.IsMeter IS NULL)
                    AND d.Name = ?
                    AND (d.KeyValue = '' OR d.KeyValue IS NULL OR UPPER(d.KeyValue) = 'ENVIRONMENT')
                    {fclause}
                    {env_clause}
                """
                rows = conn.execute(q, params).fetchall()
                if not rows:
                    return pd.DataFrame(columns=["timestamp", vname])
                df = pd.DataFrame(rows, columns=["name","vkey","units","freq","y","m","d","h","min","value"])
                # build timestamps (E+ hour = end-of-interval → shift start; Year 0 → 2002)
                y = df["y"].replace(0, 2002)
                ts = pd.to_datetime(
                    dict(year=y, month=df["m"], day=df["d"], hour=(df["h"] - 1).clip(lower=0), minute=df["min"]),
                    errors="coerce"
                )
                df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
                # label with units
                u = df["units"].iloc[0] if len(df) else ""
                col = f"{vname} [{u}]" if u else vname
                out = df[["timestamp", "value"]].rename(columns={"value": col})
                return out

            # fetch all variables and outer-join on timestamp
            wide = None
            got_any = False
            for v in variables:
                dfv = _pull_var(v)
                if dfv.empty:
                    continue
                got_any = True
                wide = dfv if wide is None else wide.merge(dfv, on="timestamp", how="outer")

            if not got_any or wide is None or wide.empty:
                hints = conn.execute("""
                    SELECT DISTINCT d.Name
                    FROM ReportData r
                    JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
                    WHERE (d.IsMeter = 0 OR d.IsMeter IS NULL)
                    AND d.Name LIKE 'Site %'
                    ORDER BY d.Name
                """).fetchall()
                names = [h[0] for h in hints][:30]
                raise ValueError(
                    "No weather-series found in SQL for the requested variables.\n"
                    f"Found in DB (sample): {names}\n"
                    "Tip: ensure Output:Variable objects exist for the site variables you need."
                )

            # sort and (optionally) resample (variables → aggregate mean by default)
            wide = wide.sort_values("timestamp")
            if resample:
                pieces = []
                for col in [c for c in wide.columns if c != "timestamp"]:
                    s = wide.set_index("timestamp")[col].resample(resample).agg(aggregate)
                    pieces.append(s.rename(col))
                wide = pd.concat(pieces, axis=1).reset_index()

            # write CSV
            out_path = os.path.join(self.out_dir, csv_filename)
            wide.to_csv(out_path, index=False)

            # summary per column
            summaries = []
            t0 = wide["timestamp"].min()
            t1 = wide["timestamp"].max()
            nrows = len(wide)
            for col in [c for c in wide.columns if c != "timestamp"]:
                series = pd.to_numeric(wide[col], errors="coerce")
                summaries.append({
                    "series": col,
                    "rows": int(series.count()),
                    "min": float(series.min()) if series.count() else None,
                    "mean": float(series.mean()) if series.count() else None,
                    "max": float(series.max()) if series.count() else None,
                })
            summary_df = pd.DataFrame(summaries)
            # prepend global window info as attrs
            summary_df.attrs["window_start"] = pd.to_datetime(t0) if pd.notna(t0) else None
            summary_df.attrs["window_end"]   = pd.to_datetime(t1) if pd.notna(t1) else None
            summary_df.attrs["rows"]         = int(nrows)

            if print_summary:
                self._log(1, f"[weather→csv] Wrote {out_path} with shape {wide.shape}")
                self._log(1, f"Window: {summary_df.attrs['window_start']} → {summary_df.attrs['window_end']} (rows={nrows})")
                try:
                    from IPython.display import display  # type: ignore
                    display(summary_df)
                except Exception:
                    self._log(1, str(summary_df.head()))

            return out_path, summary_df
        finally:
            conn.close()

    # ------------------ HVAC kill switch -----------------

    def enable_hvac_off_via_schedules(
        self,
        schedule_names: list[str] | None = None,
        *,
        autodetect: bool = True,              # if no names given, grab schedules containing "avail" or "hvac"
        verbose: bool = True
    ) -> None:
        """
        Force selected schedules to 0.0 every system timestep → disables HVAC via availability.
        Use with schedules like "HVAC Operation Schedule", "System Availability Schedule", etc.

        Example:
        util.enable_hvac_off_via_schedules(["HVAC Operation Schedule", "Central Fan Avail"])
        """
        self._hvac_kill_enabled = True
        self._hvac_kill_handles: list[int] = []
        self._hvac_kill_verbose = bool(verbose)
        wanted = set((schedule_names or []))

        def _after_warmup(s):
            handles: list[int] = []

            # Collect candidate schedule names if autodetect requested or none provided
            cand_names = set()
            if autodetect or not wanted:
                import re
                rx = re.compile(r"(avail|hvac)", re.IGNORECASE)
                for typ in self._SCHEDULE_TYPES:
                    for nm in (self.api.exchange.get_object_names(s, typ) or []):
                        if rx.search(nm):
                            cand_names.add(nm)

            targets = wanted | cand_names
            if not targets:
                if self._hvac_kill_verbose:
                    self._log(1, "[HVAC OFF] No schedule targets found.")
                return

            # Try each schedule type for each target name until a handle resolves
            for nm in sorted(targets):
                got = False
                for typ in self._SCHEDULE_TYPES:
                    try:
                        h = self.api.exchange.get_actuator_handle(s, typ, "Schedule Value", nm)
                    except Exception:
                        h = -1
                    if h != -1:
                        handles.append(h)
                        got = True
                        break
                if self._hvac_kill_verbose and not got:
                    self._log(1, f"[HVAC OFF] Could not resolve schedule actuator for '{nm}'.")

            self._hvac_kill_handles = handles
            if self._hvac_kill_verbose:
                self._log(1, f"[HVAC OFF] Resolved {len(handles)} schedule handles.")

        def _on_tick(s):
            if not getattr(self, "_hvac_kill_enabled", False) or self.api.exchange.warmup_flag(s):
                return
            for h in getattr(self, "_hvac_kill_handles", []):
                # Force schedule to zero each timestep
                self.api.exchange.set_actuator_value(s, h, 0.0)

        # Register with your unified registrar (so runs survive reset_state)
        pair_warmup = (self.api.runtime.callback_after_new_environment_warmup_complete, _after_warmup)
        pair_tick   = (self.api.runtime.callback_begin_system_timestep_before_predictor, _on_tick)

        # avoid duplicate registrations if called twice
        for pair in (pair_warmup, pair_tick):
            if pair not in self._extra_callbacks:
                self._extra_callbacks.append(pair)

        # if someone calls this *after* set_model but *before* a run, make sure callbacks are queued
        # (run_design_day/run_annual call _register_callbacks() after reset_state, so no need to call it here)

    def disable_hvac_off_via_schedules(self) -> None:
        self._hvac_kill_enabled = False
        self._hvac_kill_handles = []

    # -------- HVAC ------------

    def occupancy_handler(self, s, **overrides):
        """
        Begin-iteration handler: randomized zone occupancy via People actuators.

        Draws a truncated Poisson headcount in [min,max] for each zone at the
        start of every system timestep, then splits evenly across that zone's
        People objects.

        Register with either:
            util.register_begin_iteration(["occupancy_handler"])
        or
            util.register_begin_iteration([
                {"method_name": "occupancy_handler",
                "key_wargs": {"lam": 33.0, "min": 20, "max": 45, "seed": 123}}
            ])
        """

        # --- skip during warmup unless your registry was set to run then
        if self.api.exchange.warmup_flag(s):
            return

        # --- 1) one-time defaults (so string-only registration works)
        d = self.__dict__
        d.setdefault("_occ_rand_min", 20)
        d.setdefault("_occ_rand_max", 50)
        d.setdefault("_occ_rand_lambda", 30.0)
        d.setdefault("_occ_rand_seed", None)
        d.setdefault("_occ_rand_overrides_locked", False)  # apply kwargs once
        d.setdefault("_occ_rand_handles_ready", False)

        # --- 2) apply overrides ONCE (avoid reseeding every tick)
        if overrides and not d["_occ_rand_overrides_locked"]:
            if "min" in overrides: d["_occ_rand_min"] = int(overrides["min"])
            if "max" in overrides: d["_occ_rand_max"] = int(overrides["max"])
            if "lam" in overrides: d["_occ_rand_lambda"] = float(overrides["lam"])
            if "seed" in overrides and overrides["seed"] is not None:
                d["_occ_rand_seed"] = int(overrides["seed"])
            d["_occ_rand_overrides_locked"] = True  # prevents per-tick reseed

        # optional explicit reseed support if ever needed:
        if overrides.get("reseed", False):
            d["_occ_rand_seed"] = int(overrides.get("seed", d["_occ_rand_seed"] or 0))
            d["_occ_rng"] = np.random.default_rng(d["_occ_rand_seed"])

        # --- 3) RNG (create once; do NOT recreate each tick)
        if "_occ_rng" not in d:
            d["_occ_rng"] = np.random.default_rng(d["_occ_rand_seed"])

        # --- 4) People actuator handle resolution (once)
        if not d["_occ_rand_handles_ready"]:
            try:
                zones_live = list(self.api.exchange.get_object_names(s, "Zone") or [])
            except Exception:
                zones_live = self.list_zone_names(preferred_sources=("sql", "api", "idf"))

            z2p = self._occ_default_map_zone_to_people(s, zones_subset=zones_live, verbose=getattr(self, "_occ_verbose", True))

            handles: dict[str, list[int]] = {}
            total = 0
            for z, plist in (z2p or {}).items():
                for pname in (plist or []):
                    try:
                        h = self.api.exchange.get_actuator_handle(s, "People", "Number of People", pname)
                    except Exception:
                        h = -1
                    if h != -1:
                        handles.setdefault(z, []).append(h); total += 1

            d["_occ_rand_handles"] = handles
            d["_occ_rand_handles_ready"] = True
            if getattr(self, "_occ_verbose", True):
                if total == 0:
                    self._log(1, "[occ-counter] No People handles resolved. (No People objects or name mismatch?)")
                else:
                    self._log(1, f"[occ-counter] Resolved {total} People handles across {sum(1 for z in handles if handles[z])} zones.")

        # --- 5) sampler: truncated Poisson
        def _poisson_trunc(_rng, _lam, _lo, _hi, _max_tries=32):
            lo_i, hi_i = int(min(_lo, _hi)), int(max(_lo, _hi))
            lam_f = float(max(1e-6, _lam))
            for _ in range(_max_tries):
                k = int(_rng.poisson(lam_f))
                if lo_i <= k <= hi_i:
                    return k
            # rare fallback
            return int(np.clip(int(_rng.poisson(lam_f)), lo_i, hi_i))

        # --- 6) draw + set per zone
        lo, hi, lam = d["_occ_rand_min"], d["_occ_rand_max"], d["_occ_rand_lambda"]
        rng = d["_occ_rng"]
        handles = d.get("_occ_rand_handles", {})
        if not handles:
            return

        for z, hlist in handles.items():
            if not hlist:
                continue
            k = _poisson_trunc(rng, lam, lo, hi)
            per = float(k) / float(len(hlist))
            # set People actuators for this zone
            for h in hlist:
                self.api.exchange.set_actuator_value(s, h, per)
                # print('Occupants',per)

            # record the zone headcount once per zone (for CO₂ follower)
            if not hasattr(self, "_occ_rand_last_counts"):
                self._occ_rand_last_counts = {}
            self._occ_rand_last_counts[z] = int(k)

    def co2_set_outdoor_ppm(self, s, **opts):
        """
        ONE method: set the OUTDOOR CO₂ schedule and verify via read-back.

        Requires: prepare_run_with_co2(...) already ran (creates/binds the schedule).

        Kwargs:
        value_ppm          : float = 0.0
        clamp              : (min,max) = (0.0, 5000.0)
        log_every_minutes  : int | None = 60   # heartbeat; None = no prints
        verify             : bool = True       # read-back right after setting
        """
        if self.api.exchange.warmup_flag(s):
            return

        d = self.__dict__
        v = float(opts.get("value_ppm", 0.0))
        vmin, vmax = opts.get("clamp", (0.0, 5000.0))
        v = max(float(vmin), min(float(vmax), v))
        verify = bool(opts.get("verify", True))
        log_every = opts.get("log_every_minutes", 60)

        sched_name = getattr(self, "_co2_outdoor_schedule", None)
        if not sched_name:
            self._log(1, "[co2-outdoor] prepare_run_with_co2(...) wasn’t called for the active IDF.")
            return

        # Resolve (or re-resolve) actuator handle when state changes
        need_resolve = (
            d.get("_co2_outdoor_sched_handle", -1) == -1
            or d.get("_co2_outdoor_sched_state_id") != id(self.state)
        )
        if need_resolve:
            h = -1
            for typ in getattr(self, "_SCHEDULE_TYPES", ("Schedule:Compact","Schedule:Constant","Schedule:File","Schedule:Year")):
                try:
                    h = self.api.exchange.get_actuator_handle(s, typ, "Schedule Value", sched_name)
                except Exception:
                    h = -1
                if h != -1:
                    break
            d["_co2_outdoor_sched_handle"] = h
            d["_co2_outdoor_sched_state_id"] = id(self.state)
            if h == -1:
                self._log(1, f"[co2-outdoor] Couldn’t resolve schedule actuator for '{sched_name}'.")
                return

        # Set + optional verify
        h = d["_co2_outdoor_sched_handle"]
        self.api.exchange.set_actuator_value(s, h, v)
        # print('CO2',v)

        cur = None
        if verify:
            try:
                cur = float(self.api.exchange.get_actuator_value(s, h))
            except Exception:
                cur = None

        # Heartbeat print: once per simulated hour (or any N minutes)
        if log_every is not None:
            try:
                # minute-of-hour based on timestep (E+ hour is end-of-interval; this is good enough for logging)
                N = max(1, int(self.api.exchange.num_time_steps_in_hour(s)))
                minute = int(round((self.api.exchange.zone_time_step_number(s) - 1) * (60 / N)))
                if minute % int(log_every) == 0:
                    # avoid spamming: print only when minute changes
                    last = d.get("_co2_outdoor_last_log_minute", None)
                    if last != minute:
                        if cur is None:
                            self._log(1, f"[co2-outdoor] set={v:.1f} ppm")
                        else:
                            self._log(1, f"[co2-outdoor] set={v:.1f} ppm  read-back={cur:.1f} ppm")
                        d["_co2_outdoor_last_log_minute"] = minute
            except Exception:
                pass

    def _discover_zone_inlet_nodes_from_sql(self) -> dict[str, list[str]]:
        """
        Fast discovery of per-zone *air* inlet node keys using a single SQL pass.

        Returns: {zone: [node_key, ...]} with original key casing.
        """
        import os, sqlite3
        assert self.out_dir, "set_model(...) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            return {}

        # patterns we prefer/exclude (searched case-insensitively in SQL)
        include_any = (" IN NODE", " ATU IN NODE", " ZONE COIL AIR IN NODE")
        exclude_any = (" WATER ", "CONDENSER", " PUMP ", " BOILER ", " CHILLER",
                    " DEMAND ", " BYPASS ", " PIPE ", " OUT NODE")  # OUT NODE ~ return/exhaust

        zones = self.list_zone_names(preferred_sources=("sql","api","idf"))
        if not zones:
            return {}

        # Build LIKE predicate parts
        inc_like = " OR ".join(["UPPER(k.KeyValue) LIKE ?"] * len(include_any))
        exc_like = " AND ".join(["UPPER(k.KeyValue) NOT LIKE ?"] * len(exclude_any))
        z_like   = " OR ".join(["UPPER(k.KeyValue) LIKE ?"] * len(zones))

        # SQL: keys that have BOTH Mass Flow and Temperature (air-node hint),
        # then filter by include/exclude/name-of-zone patterns.
        q = f"""
            WITH air_keys AS (
                SELECT k.KeyValue
                FROM ReportDataDictionary k
                WHERE k.Name IN ('System Node Mass Flow Rate', 'System Node Temperature')
                AND k.KeyValue IS NOT NULL AND k.KeyValue <> ''
                GROUP BY k.KeyValue
                HAVING COUNT(DISTINCT k.Name) = 2
            )
            SELECT k.KeyValue
            FROM air_keys k
            WHERE ({inc_like})
            AND {exc_like}
            AND ({z_like})
        """
        params = (
            [f"%{p}%" for p in include_any] +
            [f"%{p}%" for p in exclude_any] +
            [f"%{z.upper()}%" for z in zones]
        )

        conn = sqlite3.connect(sql_path)
        try:
            keys = [r[0] for r in conn.execute(q, params).fetchall()]
        finally:
            conn.close()

        if not keys:
            return {}

        # Group keys by zone (keep order stable: prefer "... IN NODE" first, then ATU, then COIL AIR)
        zmap: dict[str, list[str]] = {z: [] for z in zones}
        def rank(z, key):
            ku = key.upper()
            zu = z.upper()
            if f"{zu} IN NODE" in ku: return 0
            if " ATU IN NODE" in ku:  return 1
            if " ZONE COIL AIR IN NODE" in ku: return 2
            return 9

        for z in zones:
            matches = [k for k in keys if z.upper() in k.upper()]
            matches.sort(key=lambda k: rank(z, k))
            if matches:
                zmap[z] = matches

        # Drop empties
        return {z: ks for z, ks in zmap.items() if ks}

    def probe_zone_air_and_supply(self, s, **opts):
        """
        Callback (fast): per zone snapshot of air state + supply aggregate.
        Prints at a configurable interval and returns a dict snapshot.

        Fallbacks:
        • Zone w: try Zone Air Humidity Ratio → Zone Mean Air Humidity Ratio →
                    compute from (zone T, zone RH, site P).
        • Site w: try Site Outdoor Air Humidity Ratio → compute from (To, RHo, Po).
        • Supply node w: try node Humidity Ratio → compute from (node T, node RH, site P).

        Kwargs:
        log_every_minutes: int | None = 1   # 1 => every timestep; None => no prints
        precision: int = 3
        """
        ex = self.api.exchange
        if ex.warmup_flag(s):
            return

        import math
        import numpy as _np

        d = self.__dict__
        log_every = opts.get("log_every_minutes", 1)
        prec = int(opts.get("precision", 3))

        # --- tiny psychro helper: w from T[°C], RH[%], P[Pa] (Tetens) ---
        def _w_from_T_RH_P(Tc, RH_pct, P_pa):
            try:
                Tc = float(Tc); RH_pct = float(RH_pct); P_pa = float(P_pa)
            except Exception:
                return _np.nan
            if not (_np.isfinite(Tc) and _np.isfinite(RH_pct) and _np.isfinite(P_pa) and P_pa > 1000.0):
                return _np.nan
            # Tetens saturation pressure (Pa)
            psat = 610.94 * math.exp(17.625 * Tc / (Tc + 243.04))
            pw = max(0.0, min(1.0, RH_pct / 100.0)) * psat
            denom = max(1.0, P_pa - pw)
            return 0.62198 * pw / denom  # kg/kg dry air

        # ---------- one-time REQUESTS per state (no handle resolution yet) ----------
        if d.get("_probe_req_state_id") != id(self.state):
            d["_probe_req_state_id"] = id(self.state)

            # Discover inlet nodes via SQL once (fast path)
            z2nodes = self._discover_zone_inlet_nodes_from_sql()
            if not z2nodes:
                zones = self.list_zone_names(preferred_sources=("sql","api","idf"))
                z2nodes = {z: [f"{z} IN NODE", f"{z} ATU IN NODE"] for z in zones}

            pref_single = True  # set False to keep aggregating when multiple nodes exist
            if pref_single:
                z2nodes = {
                    z: ([n for n in nodes if n.upper() == f"{z.upper()} IN NODE"] or nodes)
                    for z, nodes in z2nodes.items()
                }
            d["_probe_zone_nodes"] = z2nodes

            # Request ZONE variables (include both Air/Mean names for T, w, RH)
            for z in z2nodes:
                for nm in (
                    "Zone Mean Air Temperature", "Zone Air Temperature",
                    "Zone Air Humidity Ratio", "Zone Mean Air Humidity Ratio",
                    "Zone Air Relative Humidity", "Zone Mean Air Relative Humidity",
                    "Zone Air CO2 Concentration"
                ):
                    try: ex.request_variable(s, nm, z)
                    except Exception: pass

            # Request SITE variables (with fallbacks to compute w)
            for nm in (
                "Site Outdoor Air Drybulb Temperature",
                "Site Outdoor Air Humidity Ratio",
                "Site Outdoor Air Relative Humidity",
                "Site Outdoor Air Barometric Pressure",
                "Site Outdoor Air CO2 Concentration",
            ):
                try: ex.request_variable(s, nm, "Environment")
                except Exception: pass

            # Request NODE variables (and RH as fallback for w)
            node_vars = (
                "System Node Mass Flow Rate",
                "System Node Temperature",
                "System Node Humidity Ratio",       # may not exist on all nodes
                "System Node Relative Humidity",    # fallback to compute w
                "System Node CO2 Concentration",    # may not exist
            )
            for nodes in z2nodes.values():
                for n in nodes:
                    for nm in node_vars:
                        try: ex.request_variable(s, nm, n)
                        except Exception: pass

            # mark: handles not resolved yet
            d["_probe_handles_ready"] = False

            # CO2 schedule fallback info
            d["_probe_use_co2_sched"] = getattr(self, "_co2_outdoor_sched_handle", -1) != -1

        # ---------- resolve HANDLES once when data are fully ready ----------
        if not d.get("_probe_handles_ready", False):
            if not ex.api_data_fully_ready(s):
                return  # wait; handles not reliable yet

            def H(nm, key):
                try: return ex.get_variable_handle(s, nm, key)
                except Exception: return -1

            z2nodes = d["_probe_zone_nodes"]
            zones = list(z2nodes.keys())

            # Zone handles (T, w, RH) with Air+Mean variants
            d["_probe_h_zone_T_mean"] = {z: H("Zone Mean Air Temperature", z)      for z in zones}
            d["_probe_h_zone_T_air"]  = {z: H("Zone Air Temperature", z)            for z in zones}

            d["_probe_h_zone_w_air"]  = {z: H("Zone Air Humidity Ratio", z)         for z in zones}
            d["_probe_h_zone_w_mean"] = {z: H("Zone Mean Air Humidity Ratio", z)    for z in zones}

            d["_probe_h_zone_rh_air"]  = {z: H("Zone Air Relative Humidity", z)     for z in zones}
            d["_probe_h_zone_rh_mean"] = {z: H("Zone Mean Air Relative Humidity", z)for z in zones}

            d["_probe_h_zone_CO2"] = {z: H("Zone Air CO2 Concentration", z)         for z in zones}

            # Site handles
            d["_probe_h_site_T"]   = H("Site Outdoor Air Drybulb Temperature",   "Environment")
            d["_probe_h_site_w"]   = H("Site Outdoor Air Humidity Ratio",        "Environment")
            d["_probe_h_site_RH"]  = H("Site Outdoor Air Relative Humidity",     "Environment")
            d["_probe_h_site_P"]   = H("Site Outdoor Air Barometric Pressure",   "Environment")
            d["_probe_h_site_CO2"] = H("Site Outdoor Air CO2 Concentration",     "Environment")

            # Per-zone node handle tuples (m, T, w, RH, CO2)
            znode_handles = {}
            for z, nodes in z2nodes.items():
                tuples = []
                for n in nodes:
                    tuples.append((
                        H("System Node Mass Flow Rate",    n),
                        H("System Node Temperature",       n),
                        H("System Node Humidity Ratio",    n),
                        H("System Node Relative Humidity", n),
                        H("System Node CO2 Concentration", n),
                    ))
                znode_handles[z] = tuples
            d["_probe_znode_handles"] = znode_handles

            # one-shot debug
            for z, tuples in znode_handles.items():
                parts = []
                for (hm, hT, hw, hRH, hC), nname in zip(tuples, z2nodes[z]):
                    parts.append(
                        f"{nname}: m={'ok' if hm!=-1 else 'NA'}, "
                        f"T={'ok' if hT!=-1 else 'NA'}, "
                        f"w={'ok' if hw!=-1 else ('rh' if hRH!=-1 else 'NA')}, "
                        f"CO2={'ok' if hC!=-1 else 'NA'}"
                    )
                try:
                    self._log(1, f"[probe] {z} inlet nodes → " + ("; ".join(parts) if parts else "none"))
                except Exception:
                    pass

            d["_probe_handles_ready"] = True
            # proceed to read this same tick

        # ---------- fast READ path ----------
        def v(h):
            if h == -1: return _np.nan
            try: return float(ex.get_variable_value(s, h))
            except Exception: return _np.nan

        ts = self._occ_current_timestamp(s)

        # Site/outdoor (compute w if direct var missing)
        oT = v(d["_probe_h_site_T"])
        ow = v(d["_probe_h_site_w"])
        if not _np.isfinite(ow):
            oRH = v(d["_probe_h_site_RH"])
            oP  = v(d["_probe_h_site_P"])
            ow  = _w_from_T_RH_P(oT, oRH, oP)
        oC = v(d["_probe_h_site_CO2"])
        if (oC != oC) and d.get("_probe_use_co2_sched"):
            try: oC = float(ex.get_actuator_value(s, self._co2_outdoor_sched_handle))
            except Exception: pass
        outdoor = {"Tdb_C": oT, "w_kgperkg": ow, "co2_ppm": oC}

        # Zones
        zones_data = {}
        P_site = v(d["_probe_h_site_P"])  # for psychro fallbacks

        for z, tuples in d["_probe_znode_handles"].items():
            # Zone T: prefer Mean, then Air
            zT = v(d["_probe_h_zone_T_mean"].get(z, -1))
            if not _np.isfinite(zT):
                zT = v(d["_probe_h_zone_T_air"].get(z, -1))

            # Zone w: try Air, then Mean; else compute from RH (Air → Mean)
            zw = v(d["_probe_h_zone_w_air"].get(z, -1))
            if not _np.isfinite(zw):
                zw = v(d["_probe_h_zone_w_mean"].get(z, -1))
            if not _np.isfinite(zw):
                zRH = v(d["_probe_h_zone_rh_air"].get(z, -1))
                if not _np.isfinite(zRH):
                    zRH = v(d["_probe_h_zone_rh_mean"].get(z, -1))
                zw = _w_from_T_RH_P(zT, zRH, P_site)

            # Zone CO2
            zC = v(d["_probe_h_zone_CO2"].get(z, -1))

            # Supply (mass-flow-weighted; compute node w from RH if needed)
            m_list, T_list, w_list, C_list = [], [], [], []
            for (hm, hT, hw, hRH, hC) in tuples:
                m  = v(hm)
                if m == m:  # valid
                    Tn = v(hT)
                    wn = v(hw)
                    if not _np.isfinite(wn):
                        RHn = v(hRH)
                        wn  = _w_from_T_RH_P(Tn, RHn, P_site)
                    Cn = v(hC)
                    m_list.append(m); T_list.append(Tn); w_list.append(wn); C_list.append(Cn)

            m_sum = float(_np.nansum(m_list)) if m_list else _np.nan
            if m_list:
                wts = _np.array(m_list, dtype=float)
                def wavg(arr):
                    arr = _np.array(arr, dtype=float)
                    ok = (~_np.isnan(arr)) & (wts > 0)
                    return float(_np.average(arr[ok], weights=wts[ok])) if ok.any() else _np.nan
                sT, sw, sC = wavg(T_list), wavg(w_list), wavg(C_list)
            else:
                sT = sw = sC = _np.nan

            zones_data[z] = {
                "air":     {"Tdb_C": zT, "w_kgperkg": zw, "co2_ppm": zC},
                "supply":  {"m_dot_kgs": m_sum, "Tdb_C": sT, "w_kgperkg": sw, "co2_ppm": sC},
                "inlet_nodes": list(self._probe_zone_nodes.get(z, [])),
            }

        payload = {"timestamp": ts, "outdoor": outdoor, "zones": zones_data}

        # prints
        if log_every is not None:
            try:
                N = max(1, int(ex.num_time_steps_in_hour(s)))
                minute = int(round((ex.zone_time_step_number(s) - 1) * (60 / N)))
                last_ts = d.get("_probe_last_ts")
                if (minute % int(log_every) == 0) and (last_ts != ts):
                    d["_probe_last_ts"] = ts
                    def F(x, suf=""):
                        return f"{x:.{prec}f}{suf}" if (x==x) else "NA"
                    o = outdoor
                    self._log(1, f"[probe] {ts} | Outdoor: T={F(o['Tdb_C'],'C')} w={F(o['w_kgperkg'])} CO2={F(o['co2_ppm'],'ppm')}")
                    for z in sorted(zones_data):
                        a = zones_data[z]["air"]; su = zones_data[z]["supply"]
                        self._log(1, "         "
                            f"{z}: air(T={F(a['Tdb_C'],'C')}, w={F(a['w_kgperkg'])}, CO2={F(a['co2_ppm'],'ppm')}) | "
                            f"supply(m={F(su['m_dot_kgs'])}, T={F(su['Tdb_C'],'C')}, w={F(su['w_kgperkg'])}, CO2={F(su['co2_ppm'],'ppm')})")
            except Exception:
                pass

        self._probe_last_snapshot = payload
        return payload

    def api_check_zone_humidity_ratio(self, s, **opts):
        """
        After-HVAC callback: verify that the API returns Zone Mean Air Humidity Ratio.
        If missing/NaN, also show a fallback computed from Zone Air RH, Zone Mean T, and Site P.

        Options:
        zones: list[str] | None = None     # default: all non-plenum zones
        exclude_patterns: tuple[str,...] = ("PLENUM",)
        max_zones: int = 6                 # log at most this many zones per tick
        precision: int = 6
        log_label: str = "[w-check]"
        """
        ex = self.api.exchange
        if ex.warmup_flag(s):
            return

        import math

        d = self.__dict__
        zones_opt      = opts.get("zones", None)
        exclude_pats   = tuple(opts.get("exclude_patterns", ("PLENUM",)))
        max_zones      = int(opts.get("max_zones", 6))
        prec           = int(opts.get("precision", 6))
        LOG            = str(opts.get("log_label", "[w-check]"))

        # --- one-time: choose zones, request variables, cache handles
        if d.get("_wcheck_state_id") != id(self.state):
            d["_wcheck_state_id"] = id(self.state)

            # pick zones
            if zones_opt:
                zones = [z for z in zones_opt]
            else:
                try:
                    zones = self.list_zone_names(preferred_sources=("sql","api","idf"))
                    # exclude common plenums
                    zones = [z for z in zones if all(p not in z.upper() for p in exclude_pats)]
                except Exception:
                    zones = []
            d["_wcheck_zones"] = zones

            # request variables
            for z in zones:
                for nm in ("Zone Mean Air Humidity Ratio",
                        "Zone Air Relative Humidity",
                        "Zone Mean Air Temperature"):
                    try: ex.request_variable(s, nm, z)
                    except Exception: pass
            try: ex.request_variable(s, "Site Outdoor Air Barometric Pressure", "Environment")
            except Exception: pass

            # resolve handles (may need to wait for api_data_fully_ready)
            d["_wcheck_ready"] = False

        if not d.get("_wcheck_ready", False):
            if not ex.api_data_fully_ready(s):
                # wait until E+ says variables are ready
                return
            zones = d.get("_wcheck_zones", [])
            def H(nm, key):
                try: return ex.get_variable_handle(s, nm, key)
                except Exception: return -1
            d["_wcheck_h"] = {
                "w_mean": {z: H("Zone Mean Air Humidity Ratio", z) for z in zones},
                "rh":     {z: H("Zone Air Relative Humidity",   z) for z in zones},
                "T":      {z: H("Zone Mean Air Temperature",    z) for z in zones},
                "P":           H("Site Outdoor Air Barometric Pressure", "Environment"),
            }
            # one-shot handle status
            try:
                self._log(1, f"{LOG} handles resolved for {len(zones)} zones.")
            except Exception: pass
            d["_wcheck_ready"] = True

        H = d["_wcheck_h"]
        zones = d.get("_wcheck_zones", [])

        # safe value reader
        def v(h):
            if h in (-1, None): return float("nan")
            try:
                x = float(ex.get_variable_value(s, h))
                return x if (x == x) else float("nan")  # NaN check
            except Exception:
                return float("nan")

        # psychro fallback: w from T[°C], RH[%], P[Pa] (Tetens)
        def w_from_T_RH_P(Tc, RH_pct, P_pa):
            try:
                if not (Tc == Tc and RH_pct == RH_pct and P_pa == P_pa and P_pa > 1000.0):
                    return float("nan")
                psat = 610.94 * math.exp(17.625 * Tc / (Tc + 243.04))
                pw = max(0.0, min(1.0, RH_pct/100.0)) * psat
                denom = max(1.0, P_pa - pw)
                return 0.62198 * pw / denom
            except Exception:
                return float("nan")

        # timestamp (best-effort)
        try:
            ts = self._occ_current_timestamp(s)
        except Exception:
            ts = None

        # --- log a small sample each tick
        P = v(H["P"])  # site barometric pressure
        header = f"{LOG} timestamp={ts} (showing up to {max_zones} zones)"
        try: self._log(1, header)
        except Exception: pass

        shown = 0
        for z in zones:
            if shown >= max_zones:
                break
            w_mean = v(H["w_mean"][z])
            rh     = v(H["rh"][z])
            T      = v(H["T"][z])
            w_calc = w_from_T_RH_P(T, rh, P)

            def F(x):  # format or NA
                return (f"{x:.{prec}g}" if x == x else "NA")

            line = (f"  {z}: w_mean={F(w_mean)}  | T={F(T)}C RH={F(rh)}% P={F(P)}Pa "
                    f"| w_calc={F(w_calc)}  -> used={'w_mean' if w_mean==w_mean else ('w_calc' if w_calc==w_calc else 'NA')}")
            try: self._log(1, line)
            except Exception: pass
            shown += 1   

    # KF Implementation

    def _kf_random_walk_update(
        self,
        phi,           # (m, n)
        y,             # (m,) or (m,1)
        mu_prev,       # (n,) or (n,1)
        S_prev,        # (n, n)
        Sigma_P,       # (n, n)
        Sigma_R,       # (m, m)
        *, use_pinv=True
    ):
        """
        One-step Kalman filter update for the random-walk state model:
        mu^- = mu_prev
        S^-  = S_prev + Sigma_P
        K    = S^- phi^T (phi S^- phi^T + Sigma_R)^-1
        mu   = mu^- + K (y - phi mu^-)
        S    = (I - K phi) S^-
        yhat = phi mu

        Returns (mu_k_flat, S_k, yhat_k_flat, K).
        """
        import numpy as np

        phi     = np.asarray(phi, dtype=float)
        y       = np.asarray(y, dtype=float).reshape(-1, 1)
        mu_prev = np.asarray(mu_prev, dtype=float).reshape(-1, 1)
        S_prev  = np.asarray(S_prev, dtype=float)
        Sigma_P = np.asarray(Sigma_P, dtype=float)
        Sigma_R = np.asarray(Sigma_R, dtype=float)

        # Predict
        mu_minus = mu_prev
        S_minus  = S_prev + Sigma_P

        # Innovation
        S_innov = phi @ S_minus @ phi.T + Sigma_R
        if use_pinv:
            K = S_minus @ phi.T @ np.linalg.pinv(S_innov)
        else:
            # numerically safe solve alternative
            K = np.linalg.solve(S_innov.T, (phi @ S_minus).T).T

        # Update
        yhat_minus = phi @ mu_minus
        mu_k = mu_minus + K @ (y - yhat_minus)
        S_k  = (np.eye(mu_prev.shape[0]) - K @ phi) @ S_minus
        yhat_k = phi @ mu_k

        return mu_k.reshape(-1), S_k, yhat_k.reshape(-1), K                 

    def _kf_random_walk_update_simdkalman(
        self,
        phi,           # (m, n)
        y,             # (m,) or (m,1)
        mu_prev,       # (n,) or (n,1)
        S_prev,        # (n, n)
        Sigma_P,       # (n, n)
        Sigma_R        # (m, m)
    ):
        """
        One-step Kalman filter update using simdkalman for the random-walk model:

        x_k = I x_{k-1} + w_{k-1},   w ~ N(0, Sigma_P)
        y_k = phi_k x_k + v_k,       v ~ N(0, Sigma_R)

        Returns (mu_k_flat, S_k, yhat_k_flat, K) where K is the Kalman gain
        computed explicitly for convenience.
        """
        import numpy as np
        import simdkalman

        # coerce shapes
        phi     = np.asarray(phi, dtype=float)
        y       = np.asarray(y,   dtype=float).reshape(-1)
        mu_prev = np.asarray(mu_prev, dtype=float).reshape(-1)
        S_prev  = np.asarray(S_prev,  dtype=float)
        Q       = np.asarray(Sigma_P, dtype=float)
        R       = np.asarray(Sigma_R, dtype=float)

        m, n = phi.shape
        assert mu_prev.shape[0] == n and S_prev.shape == (n, n)
        assert y.shape[0] == m and R.shape == (m, m) and Q.shape == (n, n)

        # Random-walk: A = I
        A = np.eye(n, dtype=float)

        # Build a KF instance for this step (H = phi_k changes each tick)
        kf = simdkalman.KalmanFilter(
            state_transition   = A,   # A
            process_noise      = Q,   # Q
            observation_model  = phi, # H (time-varying, so pass current phi)
            observation_noise  = R    # R
        )

        # Time update: (mu^-, S^-)
        mu_prior, S_prior = kf.predict_next(mu_prev, S_prev)   # E[x_k|k-1], Cov[x_k|k-1]

        # Measurement update: (mu_k, S_k)
        mu_k, S_k = kf.update(mu_prior, S_prior, y)[:2]        # E[x_k|k], Cov[x_k|k]

        # Predicted observation from posterior: \hat{y}_k = H mu_k
        # (could also use kf.predict_observation(mu_k, S_k)[0])
        yhat_k = phi @ mu_k

        # Kalman gain K = S^- H^T (H S^- H^T + R)^{-1} (handy to return)
        S_innov = phi @ S_prior @ phi.T + R
        K = S_prior @ phi.T @ np.linalg.pinv(S_innov)

        return mu_k.reshape(-1), S_k, yhat_k.reshape(-1), K

    def _ekf_update(
        self,
        x_prev, P_prev,         # (n,), (n,n)
        f_x, F,                 # f(x_{k-1|k-1}) as (n,), and F_{k-1} as (n,n)
        H, Q, R,                # H_k (m,n), Q (n,n), R (m,m)
        y,                      # y_k (m,) or (m,1)
        *, use_pinv=True
    ):
        """
        One-step Extended Kalman Filter.

        Predict:
        x_prior = f_x
        P_prior = F P_prev F^T + Q
        Gain:
        K = P_prior H^T (H P_prior H^T + R)^-1
        Update:
        x_post = x_prior + K (y - H x_prior)
        P_post = (I - K H) P_prior

        Returns: (x_post, P_post, yhat_post, K)
        """
        import numpy as np

        x_prev  = np.asarray(x_prev, dtype=float).reshape(-1)
        P_prev  = np.asarray(P_prev, dtype=float)
        f_x     = np.asarray(f_x,    dtype=float).reshape(-1)
        F       = np.asarray(F,      dtype=float)
        H       = np.asarray(H,      dtype=float)
        Q       = np.asarray(Q,      dtype=float)
        R       = np.asarray(R,      dtype=float)
        y       = np.asarray(y,      dtype=float).reshape(-1)

        n = x_prev.shape[0]
        m = y.shape[0]
        assert f_x.shape[0] == n and F.shape == (n, n), "F/f_x dimension mismatch"
        assert P_prev.shape == (n, n) and Q.shape == (n, n), "P/Q dimension mismatch"
        assert H.shape == (m, n) and R.shape == (m, m), "H/R dimension mismatch"

        # Predict
        x_prior = f_x
        P_prior = F @ P_prev @ F.T + Q

        # Gain
        S_innov = H @ P_prior @ H.T + R
        if use_pinv:
            K = P_prior @ H.T @ np.linalg.pinv(S_innov)
        else:
            K = np.linalg.solve(S_innov.T, (H @ P_prior).T).T

        # Update
        yhat_prior = H @ x_prior
        x_post = x_prior + K @ (y - yhat_prior)
        P_post = (np.eye(n) - K @ H) @ P_prior
        yhat_post = H @ x_post

        return x_post.reshape(-1), P_post, yhat_post.reshape(-1), K

    def probe_zone_air_and_supply_with_kf(self, s, **opts):
        """
        Probe + KF/EKF persistence using a pluggable preparer.

        Measurement policy (generic):
        - Forward-fill outdoor and zone measurements; default 0.0 at start.
        - Zone w fallback: payload w → Zone Mean Air Humidity Ratio → w(T,RH,P_site)

        Choose the preparer with:
        kf_prepare_fn(self?, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R) -> dict for _ekf_update
        (defaults to self._kf_prepare_inputs_random_walk)

        Other opts:
        kf_sigma_P_diag, kf_sigma_R_diag, kf_init_mu, kf_init_cov_diag,
        kf_sql_table, kf_zones, kf_exclude_patterns, kf_log,
        kf_db_filename, kf_batch_size, kf_commit_every_batches,
        kf_checkpoint_every_commits, kf_journal_mode, kf_synchronous
        """
        ex = self.api.exchange
        if ex.warmup_flag(s):
            return

        import os, sqlite3, math
        import numpy as _np

        # ---- probe (pass-through) ----
        passthru_keys = {
            "kf_sigma_P_diag","kf_sigma_R_diag","kf_init_mu","kf_init_cov_diag",
            "kf_sql_table","kf_zones","kf_exclude_patterns","kf_log",
            "kf_db_filename","kf_batch_size","kf_commit_every_batches",
            "kf_checkpoint_every_commits","kf_journal_mode","kf_synchronous",
            "kf_prepare_fn"
        }
        probe_kwargs = {k:v for k,v in opts.items() if k not in passthru_keys}
        payload = self.probe_zone_air_and_supply(s, **probe_kwargs)
        if not payload or "zones" not in payload:
            return payload

        d = self.__dict__

        # ---- config ----
        Sigma_P_diag = _np.asarray(opts.get("kf_sigma_P_diag", [1e-6, 1e-3, 1e-6, 1e-6, 1e-4]), dtype=float)
        Sigma_R_diag = _np.asarray(opts.get("kf_sigma_R_diag", [0.2**2, (2e-4)**2, 30.0**2]), dtype=float)
        mu0          = _np.asarray(opts.get("kf_init_mu",      [0.0, 20.0, 0.0, 0.008, 400.0]), dtype=float)
        S0_diag      = _np.asarray(opts.get("kf_init_cov_diag",[1.0, 25.0, 1.0, 1e-3, 1e3]), dtype=float)
        table_name   = str(opts.get("kf_sql_table", "KalmanEstimates"))
        kf_zones     = opts.get("kf_zones", None)
        excl_pats    = tuple(opts.get("kf_exclude_patterns", ("PLENUM",)))
        do_log       = bool(opts.get("kf_log", True))

        db_filename  = str(opts.get("kf_db_filename", "eplusout.sql"))
        batch_size   = int(opts.get("kf_batch_size", 50))
        commit_every_batches = int(opts.get("kf_commit_every_batches", 10))
        checkpoint_every_commits = int(opts.get("kf_checkpoint_every_commits", 5))
        journal_mode = str(opts.get("kf_journal_mode", "WAL"))
        synchronous  = str(opts.get("kf_synchronous", "NORMAL"))

        Sigma_P = _np.diag(Sigma_P_diag)
        Sigma_R = _np.diag(Sigma_R_diag)  # 3x3 by construction

        # ---- pick preparer (pluggable) ----
        kf_prepare_fn = opts.get("kf_prepare_fn") or self._kf_prepare_inputs_zone_energy_model

        def _call_preparer(fn, **kw):
            """Call `fn` correctly whether it is bound (method) or free function."""
            try:
                is_bound = getattr(fn, "__self__", None) is not None
            except Exception:
                is_bound = False
            return fn(**kw) if is_bound else fn(self, **kw)

        if not callable(kf_prepare_fn):
            kf_prepare_fn = getattr(self, "_kf_prepare_inputs_zone_energy_model")

        # ---- one-time initialization (state, handles, SQL) ----
        if d.get("_kf_state_id") != id(self.state):
            d["_kf_state_id"] = id(self.state)
            d["_kf_mu"]    = {}
            d["_kf_Sigma"] = {}
            d["_kf_last_out"] = {}
            d["_kf_last_air"] = {}
            d["_kf_last_sup"] = {}

            # cache chosen zones
            if kf_zones:
                zones = [z for z in kf_zones if z in payload["zones"]]
            else:
                zones = [z for z in payload["zones"].keys() if not any(p in z.upper() for p in excl_pats)]
            d["_kf_zones_cached"] = zones

            # request extra vars for w fallback
            for z in zones:
                for nm in ("Zone Mean Air Humidity Ratio",
                        "Zone Air Relative Humidity",
                        "Zone Mean Air Temperature"):
                    try: ex.request_variable(s, nm, z)
                    except Exception: pass
            try: ex.request_variable(s, "Site Outdoor Air Barometric Pressure", "Environment")
            except Exception: pass

            d["_kf_h_ready"] = False
            d["_kf_h_wmean"] = {}
            d["_kf_h_rh"]    = {}
            d["_kf_h_T"]     = {}
            d["_kf_h_Psite"] = -1

        # resolve handles once
        if not d.get("_kf_h_ready", False) and ex.api_data_fully_ready(s):
            zones = d.get("_kf_zones_cached", [])
            def H(nm, key):
                try: return ex.get_variable_handle(s, nm, key)
                except Exception: return -1
            d["_kf_h_wmean"] = {z: H("Zone Mean Air Humidity Ratio", z) for z in zones}
            d["_kf_h_rh"]    = {z: H("Zone Air Relative Humidity",   z) for z in zones}
            d["_kf_h_T"]     = {z: H("Zone Mean Air Temperature",    z) for z in zones}
            d["_kf_h_Psite"] = H("Site Outdoor Air Barometric Pressure", "Environment")
            d["_kf_h_ready"] = True

        # ---- SQL open + schema (reused) ----
        def _ensure_sql():
            if d.get("_kf_sql_disabled"):
                return False
            if d.get("_kf_sql_conn") and d.get("_kf_sql_cur") and d.get("_kf_sql_db") == db_filename:
                return True
            try:
                assert self.out_dir, "set_model(...) first so out_dir is available."
                path = os.path.join(self.out_dir, db_filename)
                conn = sqlite3.connect(path, timeout=30.0)
                cur  = conn.cursor()
                try:
                    cur.execute(f"PRAGMA journal_mode={journal_mode};")
                    cur.execute(f"PRAGMA synchronous={synchronous};")
                except Exception:
                    pass
                # minimal base schema (state columns added dynamically on first insert)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        Timestamp TEXT NOT NULL,
                        Zone      TEXT NOT NULL,
                        y_T   REAL, y_w   REAL, y_c   REAL,
                        yhat_T REAL, yhat_w REAL, yhat_c REAL
                    )
                """)
                conn.commit()

                d["_kf_sql_conn"] = conn
                d["_kf_sql_cur"]  = cur
                d["_kf_sql_db"]   = db_filename

                d["_kf_batch"]           = []
                d["_kf_batches_written"] = 0
                d["_kf_commits"]         = 0

                d["_kf_mu_cols"]    = None   # ["mu_0","mu_1",...], set on first insert
                d["_kf_sql_insert"] = None   # INSERT stmt built on first insert
                return True
            except sqlite3.DatabaseError as e:
                try: self._log(1, f"[kf-sql] DISABLED (open): {e}")
                except Exception: pass
                d["_kf_sql_disabled"] = True
                return False

        # >>> ensure SQL is ready <<<
        if not _ensure_sql():
            return payload

        # -------- common helpers (measurement-level; shared across KFs) --------
        def _fin(x):
            try: return _np.isfinite(float(x))
            except Exception: return False

        def _v(h):
            if h in (-1, None): return float("nan")
            try:
                x = float(ex.get_variable_value(s, h))
                return x if (x == x) else float("nan")
            except Exception:
                return float("nan")

        def _w_from_T_RH_P(Tc, RH_pct, P_pa):
            try:
                if not (Tc == Tc and RH_pct == RH_pct and P_pa == P_pa and P_pa > 1000.0):
                    return float("nan")
                psat = 610.94 * math.exp(17.625 * Tc / (Tc + 243.04))
                pw = max(0.0, min(1.0, RH_pct/100.0)) * psat
                denom = max(1.0, P_pa - pw)
                return 0.62198 * pw / denom
            except Exception:
                return float("nan")

        def _ffill_out(key, val, default=0.0):
            if _fin(val):
                d["_kf_last_out"][key] = float(val)
                return float(val)
            return d["_kf_last_out"].get(key, float(default))

        def _ffill_zone(cat, z, key, val, default=0.0):
            store = d["_kf_last_air"] if cat == "air" else d["_kf_last_sup"]
            zm = store.setdefault(z, {})
            if _fin(val):
                zm[key] = float(val)
                return float(val)
            return float(zm.get(key, default))

        def _ensure_prior(zone):
            if zone not in d["_kf_mu"]:
                d["_kf_mu"][zone]    = mu0.copy().reshape(-1)
                d["_kf_Sigma"][zone] = _np.diag(S0_diag.copy())
            return d["_kf_mu"][zone].reshape(-1), d["_kf_Sigma"][zone]

        def _to_iso_ts(ts_obj) -> str:
            if ts_obj:
                return str(ts_obj).replace("T", " ")
            try:
                yr = int(self.api.exchange.year(self.state))
                m  = int(self.api.exchange.month(self.state))
                d_ = int(self.api.exchange.day_of_month(self.state))
                hh = int(self.api.exchange.hour(self.state))
                mm = int(self.api.exchange.minute(self.state))
                return f"{yr:04d}-{m:02d}-{d_:02d} {hh:02d}:{mm:02d}:00"
            except Exception:
                return ""

        def _to_num(x):
            try:
                if x is None: return None
                xv = float(x)
                if _np.isfinite(xv): return xv
            except Exception: pass
            return None

        def _ins(ts, zone_name: str, names, y_vec, yhat_vec, mu_vec):
            # --- helper to read by label ---
            def _get(_names, _vec, lbl):
                try:
                    i = _names.index(lbl)
                    return _to_num(_vec[i])
                except Exception:
                    return None

            # Build dynamic state columns + prepared INSERT if first time
            if d.get("_kf_sql_insert") is None:
                cur = d["_kf_sql_cur"]
                try:
                    schema = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
                    existing_cols = {row[1] for row in schema}
                except Exception:
                    existing_cols = set()

                try:
                    import numpy as _np
                    n_state = int(_np.size(mu_vec)) if mu_vec is not None else 0
                except Exception:
                    n_state = 0

                desired = d.get("_kf_state_col_names")  # optional list of names
                if desired is not None:
                    mu_cols = [str(c) for c in desired][:n_state]
                    while len(mu_cols) < n_state:
                        mu_cols.append(f"mu_{len(mu_cols)}")
                else:
                    mu_cols = [f"mu_{i}" for i in range(n_state)]

                for col in mu_cols:
                    if col and (col not in existing_cols):
                        try:
                            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
                        except Exception:
                            pass
                d["_kf_sql_conn"].commit()

                base_cols = ["Timestamp","Zone","y_T","y_w","y_c","yhat_T","yhat_w","yhat_c"]
                all_cols  = base_cols + mu_cols
                placeholders = [":ts", ":zone", ":y_T", ":y_w", ":y_c", ":yhat_T", ":yhat_w", ":yhat_c"] \
                            + [f":{c}" for c in mu_cols]
                d["_kf_mu_cols"]    = mu_cols
                d["_kf_sql_insert"] = f"INSERT INTO {table_name} ({', '.join(all_cols)}) VALUES ({', '.join(placeholders)})"

            # compose row
            row = {
                "ts":      _to_iso_ts(ts),
                "zone":    str(zone_name),
                "y_T":     _get(names, y_vec,    "T"),
                "y_w":     _get(names, y_vec,    "w"),
                "y_c":     _get(names, y_vec,    "CO2"),
                "yhat_T":  _get(names, yhat_vec, "T"),
                "yhat_w":  _get(names, yhat_vec, "w"),
                "yhat_c":  _get(names, yhat_vec, "CO2"),
            }
            mu_cols = d.get("_kf_mu_cols") or []
            if mu_cols:
                import numpy as _np
                mu_flat = _np.asarray(mu_vec, dtype=float).reshape(-1) if mu_vec is not None else _np.array([])
                for i, col in enumerate(mu_cols):
                    row[col] = _to_num(mu_flat[i] if i < mu_flat.size else None)

            # batch + executemany
            d["_kf_batch"].append(row)
            if len(d["_kf_batch"]) >= batch_size:
                cur = d["_kf_sql_cur"]; sql = d["_kf_sql_insert"]
                try:
                    cur.executemany(sql, d["_kf_batch"])
                    d["_kf_batch"].clear()
                    d["_kf_batches_written"] += 1
                    if d["_kf_batches_written"] % commit_every_batches == 0:
                        d["_kf_sql_conn"].commit()
                        d["_kf_commits"] += 1
                        if journal_mode.upper() == "WAL" and (d["_kf_commits"] % checkpoint_every_commits == 0):
                            try: cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                            except Exception: pass
                except sqlite3.DatabaseError as e:
                    d["_kf_sql_disabled"] = True
                    try: self._log(1, f"[kf-sql] insert disabled: {e}")
                    except Exception: pass
                    try: d["_kf_sql_conn"].close()
                    except Exception: pass
                    d["_kf_sql_conn"] = None
                    d["_kf_sql_cur"]  = None

        # -------- zone loop: build common meas, delegate to preparer + EKF --------
        if kf_zones:
            zones_use = [z for z in kf_zones if z in payload["zones"]]
        else:
            zones_use = [z for z in payload["zones"].keys() if not any(p in z.upper() for p in excl_pats)]

        ts = payload["timestamp"]
        try:
            N_steps = max(1, int(ex.num_time_steps_in_hour(s)))
            dt_h = 1.0 / N_steps
        except Exception:
            dt_h = 1.0  # safe fallback if API not ready        
        oT = _ffill_out("T",  payload["outdoor"].get("Tdb_C"),     0.0)
        ow = _ffill_out("w",  payload["outdoor"].get("w_kgperkg"), 0.0)
        oc = _ffill_out("c",  payload["outdoor"].get("co2_ppm"),   0.0)
        P_site = _v(d.get("_kf_h_Psite", -1)) if d.get("_kf_h_ready", False) else float("nan")

        for zone in zones_use:
            Z = payload["zones"][zone]

            # air
            yT = _ffill_zone("air", zone, "T", Z["air"].get("Tdb_C"), 0.0)
            yw = Z["air"].get("w_kgperkg")
            if not _fin(yw) and d.get("_kf_h_ready", False):
                w_mean = _v(d["_kf_h_wmean"].get(zone, -1))
                if _fin(w_mean):
                    yw = w_mean
                else:
                    rh  = _v(d["_kf_h_rh"].get(zone, -1))
                    Tz  = _v(d["_kf_h_T"].get(zone, -1))
                    w_c = _w_from_T_RH_P(Tz, rh, P_site)
                    if _fin(w_c): yw = w_c
            yw = _ffill_zone("air", zone, "w", yw, 0.0)
            yc = _ffill_zone("air", zone, "c", Z["air"].get("co2_ppm"), 0.0)

            # supply
            sT = _ffill_zone("sup", zone, "T", Z["supply"].get("Tdb_C"),     0.0)
            sw = _ffill_zone("sup", zone, "w", Z["supply"].get("w_kgperkg"), 0.0)
            sc = _ffill_zone("sup", zone, "c", Z["supply"].get("co2_ppm"),   0.0)

            sM = _ffill_zone("sup", zone, "m", Z["supply"].get("m_dot_kgs"), 0.0)

            # common linear observation (phi) + measurement vector y
            phi = _np.asarray([
                [ (oT - sT), 1.0, 0.0,     0.0, 0.0 ],
                [ 0.0,       0.0, (ow-sw), 1.0, 0.0 ],
                [ 0.0,       0.0, (oc-sc), 0.0, 1.0 ],
            ], dtype=float)
            y   = _np.asarray([yT, yw, yc], dtype=float)

            # prior
            mu_prev, P_prev = _ensure_prior(zone)

            meas = {
                "phi": phi, "y": y, "names": ["T","w","CO2"], "ts": ts,
                "dt": dt_h, "msa": sM,
                "To": oT, "wo": ow, "co": oc,
                "Tsa": sT, "wsa": sw, "csa": sc,
            }

            # preparer → EKF inputs
            prep = _call_preparer(
                kf_prepare_fn,
                zone=zone,
                meas=meas,
                mu_prev=mu_prev, P_prev=P_prev,
                Sigma_P=Sigma_P, Sigma_R=Sigma_R
            )

            # EKF step
            mu_k, P_k, yhat_k, K = self._ekf_update(
                prep["x_prev"], prep["P_prev"],
                prep["f_x"], prep["F"],
                prep["H"], prep["Q"], prep["R"],
                prep["y"]
            )

            # persist posterior
            d["_kf_mu"][zone]    = mu_k
            d["_kf_Sigma"][zone] = P_k

            _ins(ts, zone, ["T","w","CO2"], y, yhat_k, mu_k)

            if do_log:
                # state-agnostic logging: show first few entries
                mu_preview = ", ".join(f"{v:.4g}" for v in mu_k[:5])
                self._log(
                    1,
                    "[kf] %s %s | update | yhat_T=%.3f yhat_w=%.5f yhat_c=%.1f | mu[:5]=[%s]"
                    % (ts, zone, yhat_k[0], yhat_k[1], yhat_k[2], mu_preview)
                )

        return payload

    def _kf_prepare_inputs_zone_energy_model(
        self, *,
        zone,
        meas,          # dict with: y=[Tz, wz, cz], dt, To, wo, co, Tsa, wsa, csa, msa
        mu_prev,       # (10,) previous posterior state
        P_prev,        # (10,10) previous posterior covariance
        Sigma_P,       # (10,10) process noise
        Sigma_R        # (3,3)   measurement noise
    ):
        """
        EKF preparer for the 10-state zone energy/mass balance model:

        x = [α_o, α_s, α_e, β_o, β_s, β_e, γ_e, T_z, w_z, c_z]^T

        f(x_{k-1}) = x_{k-1} + Δt * [
            0, 0, 0, 0, 0, 0, 0,
            -(α_o + m_sa α_s) T_z + α_o T_o + m_sa α_s T_sa + α_e,
            -(β_o + m_sa β_s) w_z + β_o w_o + m_sa β_s w_sa + β_e,
            -(β_o + m_sa β_s) c_z + β_o c_o + m_sa β_s c_sa + γ_e
        ]^T

        F = I + Δt * G, nonzeros only in the last 3 rows (T_z, w_z, c_z):

        Row(T_z): d/d[α_o,α_s,α_e,β_o,β_s,β_e,γ_e,T_z,w_z,c_z] =
                    [-T_z + T_o, -m_sa T_z + m_sa T_sa, 1, 0, 0, 0, 0, -(α_o + m_sa α_s), 0, 0]

        Row(w_z): [0,0,0, -w_z + w_o, -m_sa w_z + m_sa w_sa, 1, 0, 0, -(β_o + m_sa β_s), 0]

        Row(c_z): [0,0,0, -c_z + c_o, -m_sa c_z + m_sa c_sa, 0, 1, 0, 0, -(β_o + m_sa β_s)]

        H selects (T_z, w_z, c_z).

        Required meas keys:
        y   : length-3 iterable -> [Tz_meas, wz_meas, cz_meas]
        dt  : float
        To, wo, co, Tsa, wsa, csa : floats
        msa : float (kg/s). If missing/NaN -> 0.

        Returns dict compatible with self._ekf_update(...).
        """
        import numpy as np

        # --- coerce & validate ---
        x_prev = np.asarray(mu_prev, dtype=float).reshape(-1)
        P_prev = np.asarray(P_prev, dtype=float)
        Q      = np.asarray(Sigma_P, dtype=float)
        R      = np.asarray(Sigma_R, dtype=float)

        if x_prev.size != 10:
            raise ValueError(f"expected 10-state vector, got {x_prev.size}")
        if P_prev.shape != (10, 10):
            raise ValueError(f"P_prev must be (10,10), got {P_prev.shape}")
        if Q.shape != (10, 10):
            raise ValueError(f"Sigma_P must be (10,10), got {Q.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"Sigma_R must be (3,3), got {R.shape}")

        def _get(name, *alts, default=0.0):
            for k in (name, *alts):
                if k in meas:
                    return float(meas[k])
            return float(default)

        # measurements and inputs
        y   = np.asarray(meas.get("y", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        dt  = float(meas.get("dt", 1.0))
        To  = _get("To",  "T_o",  "T_out",  default=0.0)
        wo  = _get("wo",  "w_o",  "w_out",  default=0.0)
        co  = _get("co",  "c_o",  "co_out", default=0.0)
        Tsa = _get("Tsa", "T_sa", default=0.0)
        wsa = _get("wsa", "w_sa", default=0.0)
        csa = _get("csa", "c_sa", default=0.0)
        msa = _get("msa", "m_sa", "m_dot", "m_dot_kgs", default=0.0)
        if not np.isfinite(msa):
            msa = 0.0

        # unpack for readability
        ao, aS, aE, bO, bS, bE, gE, Tz, wz, cz = x_prev

        # --- f(x_prev) = x_prev + dt * g(x_prev, u_{k-1}) ---
        g = np.zeros(10, dtype=float)
        g[7] = -(ao + msa * aS) * Tz + ao * To + msa * aS * Tsa + aE   # Tz_dot
        g[8] = -(bO + msa * bS) * wz + bO * wo + msa * bS * wsa + bE   # wz_dot
        g[9] = -(bO + msa * bS) * cz + bO * co + msa * bS * csa + gE   # cz_dot (uses γ_e)
        f_x  = x_prev + dt * g

        # --- F = I + dt * G ---
        G = np.zeros((10, 10), dtype=float)

        # Row for Tz dynamics (state index 7)
        G[7, 0] = -Tz + To                      # d/d α_o
        G[7, 1] = -msa * Tz + msa * Tsa         # d/d α_s
        G[7, 2] =  1.0                          # d/d α_e
        G[7, 7] = -(ao + msa * aS)              # d/d Tz

        # Row for wz dynamics (state index 8)
        G[8, 3] = -wz + wo                      # d/d β_o
        G[8, 4] = -msa * wz + msa * wsa         # d/d β_s
        G[8, 5] =  1.0                          # d/d β_e
        G[8, 8] = -(bO + msa * bS)              # d/d wz

        # Row for cz dynamics (state index 9)
        G[9, 3] = -cz + co                      # d/d β_o
        G[9, 4] = -msa * cz + msa * csa         # d/d β_s
        G[9, 6] =  1.0                          # d/d γ_e
        G[9, 9] = -(bO + msa * bS)              # d/d cz

        F = np.eye(10, dtype=float) + dt * G

        # --- H selects (Tz, wz, cz) ---
        H = np.zeros((3, 10), dtype=float)
        H[:, 7:] = np.eye(3)

        return dict(
            x_prev=x_prev, P_prev=P_prev,
            f_x=f_x, F=F, H=H, Q=Q, R=R,
            y=y
        )



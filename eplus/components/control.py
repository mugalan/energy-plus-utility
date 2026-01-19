import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib
from typing import List, Dict, Tuple, Optional, Sequence
import sqlite3

class ControlMixin:
    def __init__(self):
        self._log(2, "Initialized ControlMixin")


    def runtime_get_actuator(
        self,
        s,
        *,
        component_type: str | None = None,
        control_type: str | None = None,
        actuator_key: str | None = None,
        handle: int | None = None,
        allow_warmup: bool = True,
        cache_handle: bool = True,
        default: float | None = None,
        log: bool = False,
    ) -> float | None:
        """
        Lightweight getter for an EnergyPlus actuator value.

        This is a *pure helper* meant to be called from your own runtime handlers
        (i.e., functions you register via `register_handlers(...)`). It performs
        handle resolution and returns the **current actuator value** as a float when
        available, otherwise **None** (or the `default` you provide).

        You may identify an actuator by either:
        - a pre-resolved integer `handle`, **or**
        - the actuator triple `(component_type, control_type, actuator_key)`.
            Handles are resolved once `api_data_fully_ready(s)` is true, and are
            cached **per run (per active state)** if `cache_handle=True`.

        Parameters
        ----------
        s : pyenergyplus State
            The runtime state passed into your callback.
        component_type : str, optional
            E+ component type (e.g., "Schedule:Compact", "People", "Weather Data").
            Required if `handle` is not provided.
        control_type : str, optional
            E+ control type (e.g., "Schedule Value", "Number of People", "Outdoor Dry Bulb").
            Required if `handle` is not provided.
        actuator_key : str, optional
            Object/key for the actuator. May be "" or "*" for wildcard actuators,
            depending on the actuator type. Required if `handle` is not provided.
        handle : int, optional
            Pre-resolved actuator handle (fast path).
        allow_warmup : bool, default True
            If False, returns `default` during warmup/sizing periods.
        cache_handle : bool, default True
            Cache resolved handles for the current run. Handles are NOT reusable
            across runs or states.
        default : float | None, default None
            Value to return when the actuator is not available yet (e.g., before
            inputs are parsed / API data not ready / handle not found).
        log : bool, default False
            If True, emits minimal diagnostics via `self._log`.

        Returns
        -------
        float | None
            The current actuator value, or `default` (None by default) if not available.

        Notes
        -----
        - This function avoids raising when the API isn’t ready; it simply returns
        `default` until `exchange.api_data_fully_ready(s)` is true.
        - Typical schedule read:
            component_type="Schedule:Compact", control_type="Schedule Value", actuator_key="<SchedName>"
        - People example:
            component_type="People", control_type="Number of People", actuator_key="<People Object Name>"

        Examples
        --------
        Use *inside your own handler* (which you will register):

        >>> def my_controller(self, s, **_):
        ...     v = self.runtime_get_actuator(
        ...         s,
        ...         component_type="Schedule:Compact",
        ...         control_type="Schedule Value",
        ...         actuator_key="FanAvailSched",
        ...         log=False
        ...     )
        ...     if v is None:
        ...         return  # not ready yet
        ...     # use v to make a decision, set other actuators, etc.

        Ad-hoc call (only safe if you are already inside a runtime hook or otherwise
        know the API data are ready):

        >>> val = util.runtime_get_actuator(
        ...     util.state,
        ...     component_type="Weather Data",
        ...     control_type="Outdoor Dry Bulb",
        ...     actuator_key="Environment"
        ... )
        """
        ex = self.exchange

        # Respect warmup preference
        try:
            if not allow_warmup and ex.warmup_flag(s):
                return default
        except Exception:
            pass

        # Per-run cache (tied to active state id)
        d = self.__dict__
        if d.get("_act_cache_state_id") != id(self.state):
            d["_act_cache_state_id"] = id(self.state)
            d["_act_handle_cache"] = {}

        # Resolve handle if needed
        h = handle if (isinstance(handle, int) and handle >= 0) else -1
        if h == -1:
            if not all([component_type, control_type]) or actuator_key is None:
                if log:
                    try: self._log(1, "[act:get] missing actuator triple and no valid handle.")
                    except Exception: pass
                return default

            key = (str(component_type).strip(), str(control_type).strip(), str(actuator_key).strip())
            h = d["_act_handle_cache"].get(key, -1) if cache_handle else -1

            if h == -1:
                try:
                    if not ex.api_data_fully_ready(s):
                        return default
                except Exception:
                    return default
                try:
                    h = ex.get_actuator_handle(s, key[0], key[1], key[2])
                except Exception:
                    h = -1
                if h == -1:
                    if log:
                        try: self._log(1, f"[act:get] handle not found for: {key}")
                        except Exception: pass
                    return default
                if cache_handle:
                    d["_act_handle_cache"][key] = h

        # Read current value
        try:
            v = ex.get_actuator_value(s, h)
            try:
                return float(v)
            except Exception:
                # If it’s not strictly numeric, just return as-is if truthy, else default
                return default if v is None else float(v)
        except Exception as e:
            if log:
                try: self._log(1, f"[act:get] read failed: {e}")
                except Exception: pass
            return default

    def runtime_set_actuator(
        self,
        s,
        *,
        component_type: str | None = None,
        control_type: str | None = None,
        actuator_key: str | None = None,
        handle: int | None = None,
        value=0.0,
        allow_warmup: bool = False,
        clamp: tuple[float, float] | None = None,
        cache_handle: bool = True,
        log: bool = False,
    ) -> bool:
        """
        Set an EnergyPlus actuator during a simulation step (to be called from your own
        runtime handlers). Returns **True** if a value was applied, else **False**.

        Identify the actuator by either:
        • a pre-resolved integer `handle`, OR
        • the actuator triple `(component_type, control_type, actuator_key)`.
            When using the triple, the handle is resolved once `api_data_fully_ready(s)` is true
            and cached **per run/per state** if `cache_handle=True`.

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed by EnergyPlus to your callback.
        component_type, control_type, actuator_key : str, optional
            Actuator triple (e.g., "Schedule:Compact", "Schedule Value", "FanAvailSched").
            Required if `handle` is not supplied.
        handle : int, optional
            Pre-resolved actuator handle (fast path).
        value : float | int | callable, default 0.0
            Number to set, or a callable returning the number. If callable, it is invoked as
            `fn(self, s)` → `fn(s)` → `fn()` (first that works) each timestep.
        allow_warmup : bool, default False
            If False, skip setting during warmup/sizing.
        clamp : (float, float) | None, default None
            Inclusive bounds; final value is clamped to this range if provided.
        cache_handle : bool, default True
            Cache the resolved handle for (component_type, control_type, key) for this run.
        log : bool, default False
            Emit one-line diagnostics via `self._log`.

        Returns
        -------
        bool
            True if set succeeded, False otherwise (e.g., API not ready, warmup disallowed,
            handle not found, non-numeric value, or API error).

        Notes
        -----
        - Handles are valid only after inputs are parsed and **only for the current run**.
        - Typical schedule actuator:
            component_type="Schedule:Compact", control_type="Schedule Value", actuator_key="<SchedName>"
        - People count example:
            component_type="People", control_type="Number of People", actuator_key="<People Object Name>"

        Examples
        --------
        Inside your controller handler (registered via `register_handlers(...)`):

        >>> def my_controller(self, s, **_):
        ...     # Force fan availability schedule to 0
        ...     ok = self.runtime_set_actuator(
        ...         s,
        ...         component_type="Schedule:Compact",
        ...         control_type="Schedule Value",
        ...         actuator_key="FanAvailSched",
        ...         value=0.0,
        ...         log=False
        ...     )
        ...     # optionally branch on 'ok'

        With a callable value + clamping:
        >>> def occ_profile(self, s):  # returns a float
        ...     minute = int(self.exchange.minute(s))
        ...     return 5.0 if 9*60 <= minute <= 17*60 else 0.0
        ...
        >>> def set_people(self, s, **_):
        ...     self.runtime_set_actuator(
        ...         s,
        ...         component_type="People",
        ...         control_type="Number of People",
        ...         actuator_key="OpenOffice People",
        ...         value=occ_profile,
        ...         clamp=(0.0, 100.0)
        ...     )
        """
        ex = self.exchange

        # 1) Skip during warmup unless allowed
        try:
            if not allow_warmup and ex.warmup_flag(s):
                return False
        except Exception:
            pass

        # 2) Evaluate value (supports callable)
        v = value
        if callable(v):
            try:
                try:
                    v = value(self, s)
                except TypeError:
                    try:
                        v = value(s)
                    except TypeError:
                        v = value()
            except Exception:
                if log:
                    try: self._log(1, "[act:set] value function failed; skipping.")
                    except Exception: pass
                return False

        # Must be numeric
        try:
            v = float(v)
        except Exception:
            if log:
                try: self._log(1, f"[act:set] non-numeric value={v!r}; skipping.")
                except Exception: pass
            return False

        # 3) Optional clamp
        if clamp is not None and len(clamp) == 2:
            lo, hi = float(clamp[0]), float(clamp[1])
            if lo > hi:
                lo, hi = hi, lo
            v = max(lo, min(hi, v))

        # 4) Per-run handle cache
        d = self.__dict__
        if d.get("_act_cache_state_id") != id(self.state):
            d["_act_cache_state_id"] = id(self.state)
            d["_act_handle_cache"] = {}  # (comp, ctrl, key) -> handle

        # 5) Resolve handle
        h = handle if (isinstance(handle, int) and handle >= 0) else -1
        if h == -1:
            if not all([component_type, control_type]) or actuator_key is None:
                if log:
                    try: self._log(1, "[act:set] missing actuator triple and no valid handle.")
                    except Exception: pass
                return False

            key = (str(component_type).strip(), str(control_type).strip(), str(actuator_key).strip())
            h = d["_act_handle_cache"].get(key, -1) if cache_handle else -1

            if h == -1:
                try:
                    if not ex.api_data_fully_ready(s):
                        return False
                except Exception:
                    return False
                try:
                    h = ex.get_actuator_handle(s, key[0], key[1], key[2])
                except Exception:
                    h = -1
                if h == -1:
                    if log:
                        try: self._log(1, f"[act:set] handle not found for: {key}")
                        except Exception: pass
                    return False
                if cache_handle:
                    d["_act_handle_cache"][key] = h

        # 6) Apply
        try:
            ex.set_actuator_value(s, h, v)
            if log:
                try:
                    if handle is not None and handle >= 0:
                        self._log(1, f"[act:set] handle={h} -> {v:.6g}")
                    else:
                        self._log(1, f"[act:set] ({component_type} | {control_type} | {actuator_key}) -> {v:.6g}")
                except Exception:
                    pass
            return True
        except Exception as e:
            if log:
                try: self._log(1, f"[act:set] failed: {e}")
                except Exception: pass
            return False

    def tick_log_actuator(
        self,
        s,
        *,
        component_type: str | None = None,
        control_type: str | None = None,
        actuator_key: str | None = None,
        handle: int | None = None,
        allow_warmup: bool = False,
        cache_handle: bool = True,
        default: float | None = float("nan"),
        label: str | None = None,
        precision: int = 6,
        level: int = 1,
        when: str = "always",         # "always" | "on_change" | "on_resolve"
        eps: float = 1e-9,            # change threshold for "on_change"
    ) -> float | None:
        """
        Read & log an EnergyPlus actuator value at runtime (to be *registered* as a handler).

        This helper simply wraps `runtime_get_actuator(...)` and adds flexible logging:
        - **always**: log every time it runs,
        - **on_change**: log only when the value changes by more than `eps`,
        - **on_resolve**: log only the first time a valid value is obtained (e.g., after handles
        become available / API is ready).

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed by EnergyPlus to your callback.
        component_type, control_type, actuator_key : str, optional
            Actuator triple (e.g., "Schedule:Compact", "Schedule Value", "FanAvailSched").
            Required if `handle` is not supplied.
        handle : int, optional
            Pre-resolved actuator handle (fast path).
        allow_warmup : bool, default False
            If False, the read is skipped during warmup/sizing periods.
        cache_handle : bool, default True
            Cache the resolved handle for this (run, triple).
        default : float | None, default NaN
            Value to return when not available (API not ready / warmup / handle missing).
        label : str | None, default None
            Optional custom label for logs; if omitted a label is built from the triple/handle.
        precision : int, default 6
            Decimal precision for the logged value.
        level : int, default 1
            Log level passed to `self._log(level, ...)` (falls back to `print` if `_log` missing).
        when : {"always","on_change","on_resolve"}, default "always"
            Logging strategy.
        eps : float, default 1e-9
            Threshold for detecting a change under `when="on_change"`.

        Returns
        -------
        float | None
            The actuator value (or `default`), same as `runtime_get_actuator(...)`.

        Notes
        -----
        - Intended to be registered via your hook registry, e.g.:
        `register_handlers("begin", [{"method_name":"tick_log_actuator", ...}])`.
        - Uses per-run memory to detect changes/resolution events; values are tracked
        separately per actuator handle or (component, control, key) triple.

        Examples
        --------
        Log a schedule value each system timestep (begin hook):

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_log_actuator",
        ...     "kwargs": {
        ...       "component_type": "Schedule:Compact",
        ...       "control_type": "Schedule Value",
        ...       "actuator_key": "FanAvailSched",
        ...       "when": "on_change",         # only when value changes
        ...       "precision": 3
        ...     }}],
        ...   run_during_warmup=False
        ... )

        Log weather override (first resolve only):

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_log_actuator",
        ...     "kwargs": {
        ...       "component_type": "Weather Data",
        ...       "control_type": "Outdoor Dry Bulb",
        ...       "actuator_key": "Environment",
        ...       "when": "on_resolve",
        ...       "label": "Outdoor Dry Bulb (override)"
        ...     }}]
        ... )
        """
        import math

        # 1) Resolve/read via your reusable getter (no side effects)
        v = self.runtime_get_actuator(
            s,
            component_type=component_type,
            control_type=control_type,
            actuator_key=actuator_key,
            handle=handle,
            allow_warmup=allow_warmup,
            cache_handle=cache_handle,
            default=default,
            log=False,  # logging handled here
        )

        # 2) Build a stable identifier + human label for logging
        if isinstance(handle, int) and handle >= 0:
            key = ("h", int(handle))
            human = label or f"handle={handle}"
        else:
            ct = (component_type or "").strip()
            tt = (control_type or "").strip()
            ak = "" if actuator_key is None else str(actuator_key).strip()
            key = ("t", ct, tt, ak)
            human = label or f"{ct} | {tt} | {ak}"

        # 3) Per-run last-value store (for on_change/on_resolve)
        d = self.__dict__
        if d.get("_actlog_state_id") != id(self.state):
            d["_actlog_state_id"] = id(self.state)
            d["_actlog_last"] = {}  # map: key -> last_value (float or None)

        last = d["_actlog_last"].get(key, None)

        # helper to check "is valid number"
        def _is_num(x):
            try:
                xf = float(x)
                return math.isfinite(xf)
            except Exception:
                return False

        should_log = False
        if when == "always":
            should_log = True
        elif when == "on_change":
            if _is_num(v) and _is_num(last):
                should_log = abs(float(v) - float(last)) > float(eps)
            elif _is_num(v) != _is_num(last):
                # one side was NaN/None, the other is numeric → treat as change
                should_log = True
        elif when == "on_resolve":
            # previously invalid → now valid
            should_log = (not _is_num(last)) and _is_num(v)
        else:
            # unknown policy → default to always
            should_log = True

        # 4) Log
        if should_log:
            if _is_num(v):
                msg = f"[act:get] {human} = {float(v):.{int(precision)}g}"
            else:
                msg = f"[act:get] {human} = {v!r}"  # e.g., None/NaN/default
            try:
                self._log(int(level), msg)  # type: ignore[attr-defined]
            except Exception:
                print(msg)

        # 5) Update last and return
        d["_actlog_last"][key] = v
        return v

    def tick_set_actuator(
        self,
        s,
        *,
        component_type: str | None = None,
        control_type: str | None = None,
        actuator_key: str | None = None,
        handle: int | None = None,
        value=0.0,                         # number or callable -> (self,s) | (s) | ()
        allow_warmup: bool = False,
        clamp: tuple[float, float] | None = None,
        cache_handle: bool = True,
        label: str | None = None,
        precision: int = 6,
        level: int = 1,
        when: str = "success",             # "always" | "success" | "on_change"
        eps: float = 1e-9,                 # change threshold for "on_change"
        read_back: bool = False,           # optionally read & log the actual actuator value after setting
        include_timestamp: bool = True,    # include model timestamp in log line
    ) -> bool:
        """
        Set & log an EnergyPlus actuator value at runtime (designed to be *registered* as a handler).

        This helper wraps `runtime_set_actuator(...)` and adds flexible logging:
        - **always**   : log every invocation (regardless of success)
        - **success**  : log only if the set call succeeds (default)
        - **on_change**: log only when the *requested* value changes (|Δ| > `eps`) since last set
                        for this actuator (per run). The set still executes; only logging is gated.

        It accepts a numeric `value` or a callable. If callable, it will be invoked each tick as
        `fn(self, s)`, or `fn(s)`, or `fn()` (first signature that works). You may also provide
        `clamp=(lo, hi)` to bound the final value before it is applied.

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed by EnergyPlus to your callback.
        component_type, control_type, actuator_key : str, optional
            Actuator triple (e.g., "Schedule:Compact", "Schedule Value", "FanAvailSched").
            Required if `handle` is not supplied.
        handle : int, optional
            Pre-resolved actuator handle (fast path). Handles are NOT reusable across runs.
        value : float | int | callable, default 0.0
            The value to apply, or a function to compute it dynamically.
        allow_warmup : bool, default False
            If False, the setter is skipped during warmup/sizing periods.
        clamp : (float, float) | None
            Inclusive [lo, hi] clamp on the final value (after evaluating a callable).
        cache_handle : bool, default True
            Cache the resolved handle for this (state_id, component, control, key).
        label : str | None
            Optional custom label for logs; otherwise derived from the triple/handle.
        precision : int, default 6
            Decimal precision in log output.
        level : int, default 1
            Log level passed to `self._log(level, ...)` (falls back to `print` if `_log` missing).
        when : {"always","success","on_change"}, default "success"
            Logging policy (see above).
        eps : float, default 1e-9
            Change threshold for `when="on_change"`.
        read_back : bool, default False
            If True, attempts to read the actuator back (via `runtime_get_actuator`) and includes
            that in the log message (useful when E+ clamps or rewrites values internally).
        include_timestamp : bool, default True
            If True, prefixes the log line with the current model timestamp if available.

        Returns
        -------
        bool
            True if a value was applied (setter call succeeded), False otherwise.

        Notes
        -----
        - Register it with your hook registry, e.g.:
        `register_handlers("begin", [{"method_name": "tick_set_actuator", "kwargs": {...}}])`
        - This method **always attempts to set** the value. The `when` policy only affects logging.
        - For per-zone occupancy, prefer `People` → `Number of People` actuators with each zone's
        People object name as the `actuator_key`, rather than a shared occupancy schedule.

        Examples
        --------
        Force fan availability schedule to 0 each system timestep, logging only on success:

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_set_actuator",
        ...     "kwargs": {
        ...       "component_type": "Schedule:Compact",
        ...       "control_type": "Schedule Value",
        ...       "actuator_key": "FanAvailSched",
        ...       "value": 0.0,
        ...       "when": "success",
        ...       "precision": 3
        ...     }}],
        ...   run_during_warmup=False
        ... )

        Drive a zone's People actuator from a callable, clamp to [0,100], log on change:

        >>> def occ_profile(self, s):
        ...     # return a float per tick
        ...     minute = int(self.exchange.minute(s))
        ...     return 50.0 if 9*60 <= minute <= 17*60 else 0.0
        ...
        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_set_actuator",
        ...     "kwargs": {
        ...       "component_type": "People",
        ...       "control_type": "Number of People",
        ...       "actuator_key": "OpenOffice People",
        ...       "value": occ_profile,
        ...       "clamp": (0.0, 100.0),
        ...       "when": "on_change",
        ...       "read_back": True
        ...     }}],
        ...   run_during_warmup=False
        ... )
        """
        import math

        # ---------- evaluate intended value (supports callables) ----------
        v_req = value
        if callable(v_req):
            try:
                try:
                    v_req = value(self, s)
                except TypeError:
                    try:
                        v_req = value(s)
                    except TypeError:
                        v_req = value()
            except Exception:
                # don't log here yet; we'll log outcome after attempting set
                v_req = None

        try:
            v_req = float(v_req) if v_req is not None else None
        except Exception:
            v_req = None

        # optional clamp on the requested value
        if v_req is not None and clamp is not None and len(clamp) == 2:
            lo, hi = float(clamp[0]), float(clamp[1])
            if lo > hi: lo, hi = hi, lo
            v_req = max(lo, min(hi, v_req))

        # ---------- build key + human label ----------
        if isinstance(handle, int) and handle >= 0:
            key = ("h", int(handle))
            human = label or f"handle={handle}"
        else:
            ct = (component_type or "").strip()
            tt = (control_type or "").strip()
            ak = "" if actuator_key is None else str(actuator_key).strip()
            key = ("t", ct, tt, ak)
            human = label or f"{ct} | {tt} | {ak}"

        # ---------- per-run last-request store (for on_change) ----------
        d = self.__dict__
        if d.get("_actset_state_id") != id(self.state):
            d["_actset_state_id"] = id(self.state)
            d["_actset_last_req"] = {}  # key -> last requested numeric value (float or None)

        last_req = d["_actset_last_req"].get(key, None)

        # change test for logging policy (we still *apply*; this only gates logging)
        def _num(x):
            try:
                xf = float(x)
                return xf if math.isfinite(xf) else None
            except Exception:
                return None

        changed = False
        if v_req is None or last_req is None:
            changed = (v_req is not None) != (last_req is not None)  # one side numeric, the other not
        else:
            changed = abs(float(v_req) - float(last_req)) > float(eps)

        # ---------- apply via reusable setter ----------
        ok = False
        if v_req is not None:
            ok = self.runtime_set_actuator(
                s,
                component_type=component_type,
                control_type=control_type,
                actuator_key=actuator_key,
                handle=handle,
                value=v_req,                 # already evaluated (and clamped)
                allow_warmup=allow_warmup,
                clamp=None,                  # (already clamped above)
                cache_handle=cache_handle,
                log=False,                   # logging handled here
            )

        # ---------- optional read-back ----------
        v_read = None
        if read_back:
            v_read = self.runtime_get_actuator(
                s,
                component_type=component_type,
                control_type=control_type,
                actuator_key=actuator_key,
                handle=handle,
                allow_warmup=allow_warmup,
                cache_handle=cache_handle,
                default=None,
                log=False,
            )

        # ---------- timestamp (optional) ----------
        ts = ""
        if include_timestamp:
            try:
                # prefer user helper if available
                if hasattr(self, "_occ_current_timestamp"):
                    ts_val = self._occ_current_timestamp(s)
                    ts = f"{ts_val} | "
                else:
                    ex = self.exchange
                    yr = int(ex.year(s)); mo = int(ex.month(s)); dy = int(ex.day_of_month(s))
                    hh = int(ex.hour(s));  mm = int(ex.minute(s))
                    ts = f"{yr:04d}-{mo:02d}-{dy:02d} {hh:02d}:{mm:02d} | "
            except Exception:
                ts = ""

        # ---------- logging per policy ----------
        do_log = (when == "always") or (when == "success" and ok) or (when == "on_change" and changed)
        if do_log:
            if ok:
                msg = f"[act:set] {ts}{human} <- {v_req:.{int(precision)}g}"
                if read_back and (v_read is not None):
                    try:
                        msg += f" | readback={float(v_read):.{int(precision)}g}"
                    except Exception:
                        msg += f" | readback={v_read!r}"
            else:
                # include reason-ish context
                detail = "no value" if v_req is None else "set failed"
                msg = f"[act:set] {ts}{human} ({detail})"
            try:
                self._log(int(level), msg)  # type: ignore[attr-defined]
            except Exception:
                print(msg)

        # ---------- remember last requested value (per run) ----------
        d["_actset_last_req"][key] = v_req

        return bool(ok)

    def runtime_get_variable(
        self,
        s,
        *,
        name: str | None = None,
        key: str | None = None,
        handle: int | None = None,
        allow_warmup: bool = True,
        cache_handle: bool = True,
        default: float | None = None,
        log: bool = False,
    ) -> float | None:
        """
        Lightweight getter for an EnergyPlus **runtime variable** value.

        Use this inside your own runtime handlers (registered via `register_handlers(...)`).
        It safely resolves a variable handle once the API is ready, caches it *per run*,
        and returns the current numeric value. If the variable is not available yet
        (e.g., before inputs are parsed or during warmup—if disallowed), it returns
        `default` (None by default).

        You can identify a variable either by:
        • a pre-resolved integer `handle`, **or**
        • the pair `(name, key)` (e.g., `"Zone Mean Air Temperature"`, `"SPACE1-1"`).
            Handles are resolved once `exchange.api_data_fully_ready(s)` is True.

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed into your callback.
        name : str, optional
            Variable name (e.g., "Zone Mean Air Temperature", "System Node Mass Flow Rate",
            "Site Outdoor Air Drybulb Temperature"). Required if `handle` is not provided.
        key : str, optional
            Variable KeyValue (e.g., a zone name, node name, or "Environment" for site vars).
            Many site/weather variables use the key `"Environment"`. Blank `""` is valid
            for some variables. Required if `handle` is not provided.
        handle : int, optional
            Pre-resolved variable handle (fast path).
        allow_warmup : bool, default True
            If False, returns `default` during warmup/sizing periods.
        cache_handle : bool, default True
            Cache the resolved (name, key) handle for the current run. Handles are *not*
            reusable across runs/states.
        default : float | None, default None
            Value to return when the variable isn't available yet / not resolved.
        log : bool, default False
            Emit minimal diagnostics via `self._log` (or `print` fallback).

        Returns
        -------
        float | None
            Current variable value (float) when available; otherwise `default`.

        Notes
        -----
        • Resolution is attempted only after `exchange.api_data_fully_ready(s)` is True.  
        • For **site/weather** variables, the key is typically `"Environment"`.  
        • If you need units, retrieve them from RDD/MDD or `eplusout.sql`—the runtime
        variable API does not return units.

        Examples
        --------
        Read a zone temperature inside your controller:

        >>> def my_controller(self, s, **_):
        ...     tz = self.runtime_get_variable(
        ...         s, name="Zone Mean Air Temperature", key="SPACE1-1"
        ...     )
        ...     if tz is None:
        ...         return  # not ready yet
        ...     # use tz...

        Read a node flow rate:

        >>> m = self.runtime_get_variable(
        ...     s, name="System Node Mass Flow Rate", key="SPACE4-1 ZONE COIL AIR IN NODE"
        ... )

        Read outdoor drybulb:

        >>> to = self.runtime_get_variable(
        ...     s, name="Site Outdoor Air Drybulb Temperature", key="Environment"
        ... )
        """
        ex = self.exchange

        # Respect warmup preference
        try:
            if not allow_warmup and ex.warmup_flag(s):
                return default
        except Exception:
            pass

        # Per-run cache (tied to active state id)
        d = self.__dict__
        if d.get("_var_cache_state_id") != id(self.state):
            d["_var_cache_state_id"] = id(self.state)
            d["_var_handle_cache"] = {}  # (name, key) -> handle

        # Resolve handle if needed
        h = handle if (isinstance(handle, int) and handle >= 0) else -1
        if h == -1:
            if not name:
                if log:
                    try: self._log(1, "[var:get] missing variable name and no valid handle.")
                    except Exception: pass
                return default

            # Key normalization
            k0 = "" if key is None else str(key).strip()
            cache_key = (str(name).strip(), k0)

            h = d["_var_handle_cache"].get(cache_key, -1) if cache_handle else -1
            if h == -1:
                # Only safe to resolve after API data are ready
                try:
                    if not ex.api_data_fully_ready(s):
                        return default
                except Exception:
                    return default

                # Try the provided key first
                try:
                    h = ex.get_variable_handle(s, cache_key[0], cache_key[1])
                except Exception:
                    h = -1

                # Helpful fallback for site/weather vars: also try "Environment" if user
                # passed blank and name looks site-like.
                if h == -1 and (not k0 or k0 == "*"):
                    nm_low = cache_key[0].lower()
                    if any(tok in nm_low for tok in ("site ", "weather", "outdoor", "environment")):
                        try:
                            h = ex.get_variable_handle(s, cache_key[0], "Environment")
                            if h != -1:
                                cache_key = (cache_key[0], "Environment")
                        except Exception:
                            h = -1

                if h == -1:
                    if log:
                        try: self._log(1, f"[var:get] handle not found for: {(cache_key[0], cache_key[1])}")
                        except Exception: pass
                    return default

                if cache_handle:
                    d["_var_handle_cache"][cache_key] = h

        # Read current value
        try:
            v = ex.get_variable_value(s, h)
            try:
                return float(v)
            except Exception:
                # In practice variable values are numeric; if not, return default
                return default
        except Exception as e:
            if log:
                try: self._log(1, f"[var:get] read failed: {e}")
                except Exception: pass
            return default    

    def tick_log_variable(
        self,
        s,
        *,
        name: str | None = None,
        key: str | None = None,
        handle: int | None = None,
        allow_warmup: bool = False,
        cache_handle: bool = True,
        default: float | None = float("nan"),
        label: str | None = None,
        precision: int = 6,
        level: int = 1,
        when: str = "always",          # "always" | "on_change" | "on_resolve"
        eps: float = 1e-9,             # change threshold for "on_change"
        include_timestamp: bool = False,
    ) -> float | None:
        """
        Read & log an EnergyPlus **runtime variable** value (to be *registered* as a handler).

        This helper wraps `runtime_get_variable(...)` and adds flexible logging policies:
        • **always**     → log every tick  
        • **on_change**  → log only when the value changes by more than `eps`  
        • **on_resolve** → log the first time a valid value is obtained (e.g., after
            handles become available / API data are ready)

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed by EnergyPlus to your callback.
        name, key : str, optional
            Variable identifier pair (e.g., name="Zone Mean Air Temperature", key="SPACE1-1").
            Required if `handle` is not supplied. For site/weather variables the key is
            typically "Environment".
        handle : int, optional
            Pre-resolved variable handle (fast path).
        allow_warmup : bool, default False
            If False, the read is skipped during warmup/sizing periods.
        cache_handle : bool, default True
            Cache the resolved handle for this (run, name, key).
        default : float | None, default NaN
            Value to return when not available (API not ready / warmup / handle missing).
        label : str | None, default None
            Optional custom label for logs; if omitted a label is built from the pair/handle.
        precision : int, default 6
            Decimal precision for the logged value.
        level : int, default 1
            Log level passed to `self._log(level, ...)` (falls back to `print` if `_log` missing).
        when : {"always","on_change","on_resolve"}, default "always"
            Logging strategy.
        eps : float, default 1e-9
            Threshold for detecting a change under `when="on_change"`.
        include_timestamp : bool, default False
            If True, prepend the current simulation timestamp (YYYY-MM-DD HH:MM).

        Returns
        -------
        float | None
            The variable value (or `default`), same as `runtime_get_variable(...)`.

        Notes
        -----
        • Intended to be registered via your hook registry, e.g.:
        `register_handlers("begin", [{"method_name":"tick_log_variable", ...}])`.  
        • Uses per-run memory to detect changes/resolution events; values are tracked
        separately per variable handle or `(name, key)` pair.

        Examples
        --------
        Log a zone temperature each system timestep (only when it changes):

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_log_variable",
        ...     "kwargs": {
        ...       "name": "Zone Mean Air Temperature",
        ...       "key": "SPACE1-1",
        ...       "when": "on_change",
        ...       "precision": 3,
        ...       "include_timestamp": True
        ...     }}],
        ...   run_during_warmup=False
        ... )

        Log outdoor drybulb (first resolve only):

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_log_variable",
        ...     "kwargs": {
        ...       "name": "Site Outdoor Air Drybulb Temperature",
        ...       "key": "Environment",
        ...       "when": "on_resolve",
        ...       "label": "Toa (Environment)"
        ...     }}]
        ... )
        """
        import math

        # 1) Resolve/read via reusable getter (no side effects)
        v = self.runtime_get_variable(
            s,
            name=name,
            key=key,
            handle=handle,
            allow_warmup=allow_warmup,
            cache_handle=cache_handle,
            default=default,
            log=False,  # logging handled here
        )

        # 2) Stable identifier + human label for logging
        if isinstance(handle, int) and handle >= 0:
            ident = ("vh", int(handle))
            human = label or f"var_handle={handle}"
        else:
            nm = (name or "").strip()
            ky = "" if key is None else str(key).strip()
            ident = ("vk", nm, ky)
            human = label or f"{nm} | {ky}"

        # 3) Per-run last-value store (for on_change/on_resolve)
        d = self.__dict__
        if d.get("_varlog_state_id") != id(self.state):
            d["_varlog_state_id"] = id(self.state)
            d["_varlog_last"] = {}  # map: ident -> last_value (float or None)

        last = d["_varlog_last"].get(ident, None)

        def _is_num(x):
            try:
                xf = float(x)
                return math.isfinite(xf)
            except Exception:
                return False

        should_log = False
        if when == "always":
            should_log = True
        elif when == "on_change":
            if _is_num(v) and _is_num(last):
                should_log = abs(float(v) - float(last)) > float(eps)
            elif _is_num(v) != _is_num(last):
                should_log = True
        elif when == "on_resolve":
            should_log = (not _is_num(last)) and _is_num(v)
        else:
            should_log = True

        # Optional timestamp (best-effort)
        ts_prefix = ""
        if include_timestamp:
            try:
                # prefer your class helper if present
                if hasattr(self, "_occ_current_timestamp"):
                    ts = self._occ_current_timestamp(s)  # type: ignore[attr-defined]
                    ts_prefix = f"{ts} | "
                else:
                    ex = self.exchange
                    yr = int(ex.year(s)); mo = int(ex.month(s)); dy = int(ex.day_of_month(s))
                    hh = int(ex.hour(s)); mm = int(ex.minute(s))
                    ts_prefix = f"{yr:04d}-{mo:02d}-{dy:02d} {hh:02d}:{mm:02d} | "
            except Exception:
                ts_prefix = ""

        # 4) Log
        if should_log:
            if _is_num(v):
                msg = f"{ts_prefix}[var:get] {human} = {float(v):.{int(precision)}g}"
            else:
                msg = f"{ts_prefix}[var:get] {human} = {v!r}"
            try:
                self._log(int(level), msg)  # type: ignore[attr-defined]
            except Exception:
                print(msg)

        # 5) Update last and return
        d["_varlog_last"][ident] = v
        return v   

    def runtime_get_meter(
        self,
        s,
        *,
        name: str | None = None,
        index: int | None = None,
        which: str = "value",        # "value" | "last" | "accum"
        allow_warmup: bool = True,
        cache_index: bool = True,
        default: float | None = None,
        log: bool = False,
    ) -> float | None:
        """
        Lightweight getter for an EnergyPlus **meter** value (for use *inside* runtime callbacks).

        Identify the meter by either:
        • `index` (pre-resolved meter index), or
        • `name` (e.g., "Electricity:Facility"). Names are case-insensitive in E+.

        The `which` selector attempts, in order of availability:
        - "value":      ex.get_meter_value(state, index)                   ← current/tick value
        - "last":       ex.get_meter_value_last_timestep(state, index)     ← previous tick (if supported)
                        (falls back to "value" if unavailable)
        - "accum":      ex.get_meter_accumulated_value(state, index)       ← cumulative since env start (if supported)
                        (falls back to "value" if unavailable)

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state provided to your callback.
        name : str, optional
            Meter name (e.g., "Electricity:Facility"). Required if `index` not provided.
        index : int, optional
            Pre-resolved meter index (fast path).
        which : {"value","last","accum"}, default "value"
            Which value to fetch (see above).
        allow_warmup : bool, default True
            If False, return `default` during warmup/sizing.
        cache_index : bool, default True
            Cache (name -> index) for this run; indexes are NOT reusable across runs/states.
        default : float | None, default None
            Returned when the meter is not yet available or resolution fails.
        log : bool, default False
            Emit minimal diagnostics via `self._log`.

        Returns
        -------
        float | None
            The meter value (usually Joules for energy meters) or `default`.

        Notes
        -----
        • Safe to call repeatedly every timestep; uses a per-run cache.
        • Doesn’t raise if API data aren’t ready; simply returns `default` until ready.
        • No unit conversion here (e.g., to kWh) — log raw, convert elsewhere if needed.
        """
        ex = self.exchange

        # Warmup?
        try:
            if not allow_warmup and ex.warmup_flag(s):
                return default
        except Exception:
            pass

        # Per-run cache (by active state id)
        d = self.__dict__
        if d.get("_meter_cache_state_id") != id(self.state):
            d["_meter_cache_state_id"] = id(self.state)
            d["_meter_index_cache"] = {}  # name(lower) -> index

        # Resolve index if needed
        idx = index if (isinstance(index, int) and index >= 0) else -1
        if idx < 0:
            if not name:
                if log:
                    try: self._log(1, "[meter:get] missing meter name and index.")
                    except Exception: pass
                return default

            key = str(name).strip().lower()
            idx = d["_meter_index_cache"].get(key, -1) if cache_index else -1

            if idx < 0:
                # Only resolve after API data are ready
                try:
                    if not ex.api_data_fully_ready(s):
                        return default
                except Exception:
                    return default
                try:
                    # Primary API
                    idx = ex.get_meter_index(s, name)
                except Exception:
                    idx = -1
                if idx < 0:
                    if log:
                        try: self._log(1, f"[meter:get] index not found for: {name!r}")
                        except Exception: pass
                    return default
                if cache_index:
                    d["_meter_index_cache"][key] = idx

        # Read value per selection
        try:
            if which == "last":
                # Try "last timestep" if available; fall back to current value
                try:
                    v = ex.get_meter_value_last_timestep(s, idx)
                except Exception:
                    v = ex.get_meter_value(s, idx)
            elif which == "accum":
                # Try "accumulated" if available; fall back to current value
                try:
                    v = ex.get_meter_accumulated_value(s, idx)
                except Exception:
                    v = ex.get_meter_value(s, idx)
            else:
                v = ex.get_meter_value(s, idx)

            try:
                return float(v)
            except Exception:
                return default if v is None else float(v)
        except Exception as e:
            if log:
                try: self._log(1, f"[meter:get] read failed ({which}): {e}")
                except Exception: pass
            return default  

    def tick_log_meter(
        self,
        s,
        *,
        name: str | None = None,
        index: int | None = None,
        which: str = "value",          # "value" | "last" | "accum"
        allow_warmup: bool = False,
        cache_index: bool = True,
        default: float | None = float("nan"),
        label: str | None = None,
        precision: int = 6,
        level: int = 1,
        when: str = "always",          # "always" | "on_change" | "on_resolve"
        eps: float = 1e-9,             # change threshold for "on_change"
        include_timestamp: bool = False,
    ) -> float | None:
        """
        Read & log an EnergyPlus **meter** value at runtime (register this as a handler).

        Wraps `runtime_get_meter(...)` and supports flexible logging policies:
        • **always**     – log every tick
        • **on_change**  – log only when the value changes by > `eps`
        • **on_resolve** – log the first time a valid value is obtained

        Parameters
        ----------
        s : pyenergyplus State
            Runtime state passed by EnergyPlus to your callback.
        name : str, optional
            Meter name (e.g., "Electricity:Facility"). Required if `index` not supplied.
        index : int, optional
            Pre-resolved meter index.
        which : {"value","last","accum"}, default "value"
            Which meter value to fetch (current / last timestep / accumulated if available).
        allow_warmup : bool, default False
            If False, skip logging during warmup/sizing periods.
        cache_index : bool, default True
            Cache name→index during this run.
        default : float | None, default NaN
            Returned (and optionally logged) when not available yet.
        label : str | None, default None
            Custom label for logs; default builds from name/index + `which`.
        precision : int, default 6
            Decimal precision for numeric log output.
        level : int, default 1
            Log level for `self._log(level, ...)` (falls back to `print`).
        when : {"always","on_change","on_resolve"}, default "always"
            Logging policy.
        eps : float, default 1e-9
            Change threshold used with `when="on_change"`.
        include_timestamp : bool, default False
            Prepend the simulation timestamp (best-effort).

        Returns
        -------
        float | None
            The meter value (or `default`) — same as `runtime_get_meter(...)`.

        Examples
        --------
        Log Facility electricity each tick (on change only), with timestamps:

        >>> util.register_handlers(
        ...   "begin",
        ...   [{"method_name": "tick_log_meter",
        ...     "kwargs": {
        ...       "name": "Electricity:Facility",
        ...       "which": "value",
        ...       "when": "on_change",
        ...       "precision": 3,
        ...       "include_timestamp": True
        ...     }}]
        ... )
        """
        import math

        # 1) Resolve/read
        v = self.runtime_get_meter(
            s,
            name=name,
            index=index,
            which=which,
            allow_warmup=allow_warmup,
            cache_index=cache_index,
            default=default,
            log=False,  # logging handled here
        )

        # 2) Build identifier + human label
        if isinstance(index, int) and index >= 0:
            ident = ("mi", int(index), which)
            human = label or f"meter_index={index} ({which})"
        else:
            nm = (name or "").strip()
            ident = ("mn", nm.lower(), which)
            human = label or f"{nm} ({which})"

        # 3) Per-run last-value store
        d = self.__dict__
        if d.get("_meterlog_state_id") != id(self.state):
            d["_meterlog_state_id"] = id(self.state)
            d["_meterlog_last"] = {}  # ident -> last_value

        last = d["_meterlog_last"].get(ident, None)

        def _is_num(x):
            try:
                xf = float(x)
                return math.isfinite(xf)
            except Exception:
                return False

        should_log = False
        if when == "always":
            should_log = True
        elif when == "on_change":
            if _is_num(v) and _is_num(last):
                should_log = abs(float(v) - float(last)) > float(eps)
            elif _is_num(v) != _is_num(last):
                should_log = True
        elif when == "on_resolve":
            should_log = (not _is_num(last)) and _is_num(v)
        else:
            should_log = True

        # Optional timestamp
        ts_prefix = ""
        if include_timestamp:
            try:
                if hasattr(self, "_occ_current_timestamp"):
                    ts = self._occ_current_timestamp(s)  # type: ignore[attr-defined]
                    ts_prefix = f"{ts} | "
                else:
                    ex = self.exchange
                    yr = int(ex.year(s)); mo = int(ex.month(s)); dy = int(ex.day_of_month(s))
                    hh = int(ex.hour(s)); mm = int(ex.minute(s))
                    ts_prefix = f"{yr:04d}-{mo:02d}-{dy:02d} {hh:02d}:{mm:02d} | "
            except Exception:
                ts_prefix = ""

        # 4) Log
        if should_log:
            if _is_num(v):
                msg = f"{ts_prefix}[meter:get] {human} = {float(v):.{int(precision)}g}"
            else:
                msg = f"{ts_prefix}[meter:get] {human} = {v!r}"
            try:
                self._log(int(level), msg)  # type: ignore[attr-defined]
            except Exception:
                print(msg)

        # 5) Update last + return
        d["_meterlog_last"][ident] = v
        return v           
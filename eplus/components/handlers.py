from typing import List, Tuple, Callable

class HandlersMixin:
    _SCHEDULE_TYPES = (
        "Schedule:Compact",
        "Schedule:Constant",
        "Schedule:File",
        "Schedule:Year",
    )
    
    def __init__(self):
        self._log(2, "Initialized HandlersMixin")
        self._runtime_log_enabled: bool = False
        self._runtime_log_func: Callable = None
        self._extra_callbacks: List[Tuple[Callable, Callable]] = []

        self.callback_aliases = {
            "begin":        "callback_begin_system_timestep_before_predictor",
            "before_hvac":  "callback_after_predictor_before_hvac_managers",
            "inside_iter":  "callback_inside_system_iteration_loop",
            "after_hvac":   "callback_end_system_timestep_after_hvac_reporting",
            "after_zone":   "callback_end_zone_timestep_after_zone_reporting",
            "after_warmup": "callback_after_new_environment_warmup_complete",
            "after_get_input": "callback_after_component_get_input",
        }

    def _register_callbacks(self):
        """Register the single timestep callback (and any extras)."""

        # TODO:Comeback to this when Understand
        # CSV occupancy tick (if enabled)
        # if getattr(self, "_occ_enabled", False) and self._occ_df is not None:
        #     self.runtime.callback_begin_system_timestep_before_predictor(
        #         self.state, self._occ_cb_tick
        #     )

        # Any extra user-queued callbacks
        for registrar, func in getattr(self, "_extra_callbacks", []):
            registrar(self.state, func)

        # Re-attach runtime logger if enabled (EnergyPlus 25.1 uses 1-arg signature)
        if getattr(self, "_runtime_log_enabled", False) and getattr(self, "_runtime_log_func", None):
            self.runtime.callback_message(self.state, self._runtime_log_func)
            # --- Generi Callback registering hub ---

    def _init_callback_hub(self):
        """
        One-time init for the generic callback hub.
        self._cb_registries: dict keyed by 'hook_key' (str), each value:
            {
            'enabled': bool,
            'run_during_warmup': bool,
            'names': [str],                # ordered names
            'specs': [(callable, dict)],   # ordered callables + kwargs
            'dispatcher': callable,        # closure(state) -> dispatch
            'runtime_register': callable,  # runtime.callback_*
            'registered_once': bool,       # whether we bound dispatcher already
            }
        """
        if not hasattr(self, "_cb_registries"):
            self._cb_registries = {}
        if not hasattr(self, "_extra_callbacks"):
            self._extra_callbacks = []  # used by your class to rebind on reset

    def _resolve_runtime_register(self, hook) -> tuple[str, callable]:
        """
        Accepts:
        - a string alias from callback_aliases
        - a full runtime attribute name (string)
        - a direct callable (runtime.callback_*)
        Returns (hook_key, runtime_register_callable).
        """
        rt = self.runtime
        if callable(hook):
            # Try to derive a stable key name from the func name
            key = getattr(hook, "__name__", "custom_callback")
            return key, hook
        if not isinstance(hook, str):
            raise TypeError(f"hook must be a string alias/name or a callable; got {type(hook)}")

        attr = self.callback_aliases.get(hook, hook)  # allow direct attr names too
        fn = getattr(rt, attr, None)
        if not callable(fn):
            raise AttributeError(f"EnergyPlus runtime has no callable '{attr}'")
        return hook, fn

    def _get_or_init_hook_registry(self, hook) -> tuple[str, dict]:
        """
        Ensure a registry for the given hook exists. Create dispatcher lazily.
        Returns (hook_key, registry_dict).
        """
        self._init_callback_hub()
        hook_key, runtime_register = self._resolve_runtime_register(hook)

        reg = self._cb_registries.get(hook_key)
        if reg is None:
            # Create a per-hook dispatcher closure
            def make_dispatcher(hkey):
                def _dispatcher(state):
                    ex = self.exchange
                    cfg = self._cb_registries.get(hkey, {})
                    if not cfg.get("enabled", True):
                        return
                    if (not cfg.get("run_during_warmup", False)) and ex.warmup_flag(state):
                        return
                    # snapshot to avoid mutation during iteration
                    for func, kw in list(cfg.get("specs", [])):
                        try:
                            func(state, **kw)
                        except Exception as e:
                            try:
                                nm = getattr(func, "__name__", "handler")
                                self._log(1, f"[{hkey}] {nm} failed: {e}")
                            except Exception:
                                pass
                return _dispatcher

            disp = make_dispatcher(hook_key)
            reg = {
                "enabled": True,
                "run_during_warmup": False,
                "names": [],
                "specs": [],
                "dispatcher": disp,
                "runtime_register": runtime_register,
                "registered_once": False,
            }
            self._cb_registries[hook_key] = reg
        else:
            # if the caller passed a different runtime_register callable, update it
            reg["runtime_register"] = runtime_register

        return hook_key, reg

    def _ensure_hook_registered(self, hook):
        """
        Bind the dispatcher to the EnergyPlus state for this hook, only once.
        Also track in _extra_callbacks so a future reset can re-bind automatically.
        """
        hook_key, reg = self._get_or_init_hook_registry(hook)
        # remember the pair so your reset routine can re-register later
        pair = (reg["runtime_register"], reg["dispatcher"])
        if pair not in self._extra_callbacks:
            self._extra_callbacks.append(pair)

        if getattr(self, "state", None) and not reg.get("registered_once", False):
            try:
                reg["runtime_register"](self.state, reg["dispatcher"])
                reg["registered_once"] = True
            except Exception:
                pass

    # --- public generic API ---
    def register_handlers(self, hook, methods, *, clear: bool = False,
                        enable: bool = True, run_during_warmup: bool | None = None) -> list[str]:
        """
        Register one or more **bound instance methods** to run at a specific
        EnergyPlus runtime **callback hook** (e.g., *begin system timestep*,
        *before HVAC managers*, *after HVAC reporting*, etc.).

        This API lets you compose a **runtime control loop** in pure Python:
        you choose *when* your code runs (the hook) and *what* runs (a list of
        instance methods, each with optional kwargs). Under the hood, a single
        dispatcher is attached to the EnergyPlus callback and invokes your
        methods in-order every time that hook fires.

        Parameters
        ----------
        hook : str | callable
            Where to attach the dispatcher. Accepts:
            - **Alias string** (recommended):

            ========  ==============================================
            alias     EnergyPlus runtime callback attribute
            --------  ----------------------------------------------
            "begin"   callback_begin_system_timestep_before_predictor
            "before_hvac"
                        callback_after_predictor_before_hvac_managers
            "inside_iter"
                        callback_inside_system_iteration_loop
            "after_hvac"
                        callback_end_system_timestep_after_hvac_reporting
            "after_zone"
                        callback_end_zone_timestep_after_zone_reporting
            "after_warmup"
                        callback_after_new_environment_warmup_complete
            "after_get_input"
                        callback_after_component_get_input
            ========  ==============================================

            - **Full attribute name** as a string, e.g.
            ``"callback_after_predictor_before_hvac_managers"``.
            - **The registration callable itself**, e.g.
            ``runtime.callback_inside_system_iteration_loop``.
        methods : Sequence[str | dict]
            Handler specs. Each handler must be a **method on this instance**
            and is invoked as: ``method(self, state, **kwargs)``.

            Acceptable forms (order matters):
            - ``"method_name"``                       → no kwargs
            - ``{"method_name": "...", "kwargs": {...}}``

            For convenience, the kwargs key also accepts:
            ``"params"``, ``"key_kwargs"``, or even the misspelling
            ``"key_wargs"`` (all treated the same).
        clear : bool, default False
            If True, **drop any previously registered handlers for this hook**
            before adding the new set.
        enable : bool, default True
            If False, keep the registry but **pause dispatch** for this hook
            (handlers remain registered and can be re-enabled later).
        run_during_warmup : bool | None, default None
            Warmup dispatch policy for this hook:
            - True  → run handlers **during warmup**;
            - False → **skip during warmup**;
            - None  → leave the current setting unchanged.

        Returns
        -------
        list[str]
            The **ordered list of registered method names** for this hook
            *after* the update.

        Behavior & Ordering
        -------------------
        - Handlers are **de-duplicated by method name**; if the same method
        appears multiple times, the **last occurrence wins** (its kwargs
        are kept).
        - Final order preserves any **existing kept** handlers, then appends
        **new names in the order provided**.
        - A single dispatcher is attached to the chosen runtime hook; later
        calls only update the internal registry and flags.
        - The dispatcher checks the EnergyPlus warmup flag each tick and
        abides by ``run_during_warmup`` for this hook.

        Side Effects
        ------------
        - Ensures the underlying EnergyPlus callback is **registered on the
        current state** so the dispatcher will fire in the next run.
        - If your workflow replaces the E+ state (e.g., via ``reset_state``),
        this class re-applies all previously attached dispatchers.

        Raises
        ------
        TypeError
            If a handler spec is not ``str`` or ``dict``, if kwargs is not a
            ``dict``, or if ``hook`` is neither a string nor a callable.
        ValueError
            If a dict spec is missing a method name or it is empty/blank.
        AttributeError
            - If a named method does not exist on this instance or is not callable.
            - If a string hook cannot be resolved to a valid runtime callback.

        Examples
        --------
        Basic: run two methods at the **beginning of each system timestep**:

        >>> util.register_handlers(
        ...     "begin",
        ...     [
        ...         "occupancy_handler",
        ...         {"method_name": "co2_set_outdoor_ppm",
        ...          "kwargs": {"value_ppm": 450.0, "log_every_minutes": None}}
        ...     ],
        ...     run_during_warmup=False
        ... )

        Use a **full attribute name** for the hook:

        >>> util.register_handlers(
        ...     "callback_end_system_timestep_after_hvac_reporting",
        ...     ["probe_zone_air_and_supply"]
        ... )

        Pass the **registration callable** directly (no strings involved):

        >>> util.register_handlers(
        ...     util.runtime.callback_inside_system_iteration_loop,
        ...     [{"method_name": "my_iterative_controller", "params": {"gain": 0.2}}]
        ... )

        Temporarily **pause** a hook without changing its handlers:

        >>> util.register_handlers("begin", [], enable=False)

        Replace existing handlers and **update kwargs** (de-dup by name, last wins):

        >>> util.register_handlers(
        ...     "after_hvac",
        ...     [
        ...         {"method_name": "probe_zone_air_and_supply_with_kf",
        ...          "kwargs": {"kf_db_filename": "kf.sqlite", "kf_batch_size": 100}},
        ...         {"method_name": "probe_zone_air_and_supply_with_kf",
        ...          "kwargs": {"kf_db_filename": "kf_big.sqlite", "kf_batch_size": 250}}
        ...     ],
        ...     clear=True
        ... )
        # → only one entry remains for 'probe_zone_air_and_supply_with_kf' with batch_size=250

        Practical pattern: **controls early, analytics late**:

        >>> util.register_handlers(
        ...     "begin",
        ...     [{"method_name": "zone_pid_controller",
        ...       "kwargs": {"setpoint_C": 22.0, "humidity_w": 0.008}}],
        ...     clear=True
        ... )
        >>> util.register_handlers(
        ...     "after_hvac",
        ...     [{"method_name": "probe_zone_air_and_supply_with_kf",
        ...       "kwargs": {"kf_db_filename": "eplusout_kf.sqlite"}}],
        ...     clear=True
        ... )

        Notes
        -----
        - Handler signature is **``handler(self, state, **kwargs)``**. If your
        logic requires resolved API handles, consider waiting for
        ``exchange.api_data_fully_ready(state)`` or registering at
        ``"after_get_input"`` / ``"after_warmup"`` to initialize and cache them.
        - Keep handlers **fast and non-blocking**—they execute in the simulation
        loop at every hook invocation.
        """
        hook_key, reg = self._get_or_init_hook_registry(hook)

        if clear:
            reg["names"].clear()
            reg["specs"].clear()

        def _extract(item):
            if isinstance(item, str):
                name, kwargs = item.strip(), {}
            elif isinstance(item, dict):
                name = str(item.get("method_name") or item.get("name") or "").strip()
                kwargs = (item.get("key_wargs")
                        or item.get("kwargs")
                        or item.get("key_kwargs")
                        or item.get("params")
                        or {})
                if not isinstance(kwargs, dict):
                    raise TypeError(f"kwargs for '{name}' must be a dict")
            else:
                raise TypeError(f"Unsupported method spec: {item!r}")

            if not name:
                raise ValueError(f"Invalid method name in spec: {item!r}")

            func = getattr(self, name, None)
            if func is None or not callable(func):
                raise AttributeError(f"No callable '{name}' found on {self.__class__.__name__}")
            return name, func, dict(kwargs)

        # dedupe (last wins)
        seen = {nm: (fn, kw) for nm, (fn, kw) in zip(reg["names"], reg["specs"])}
        for item in methods:
            nm, fn, kw = _extract(item)
            seen[nm] = (fn, kw)

        # ordered names: keep existing order for kept names, then append new
        ordered = []
        for nm in reg["names"]:
            if nm in seen and nm not in ordered:
                ordered.append(nm)
        for item in methods:
            nm = item if isinstance(item, str) else (item.get("method_name") or item.get("name"))
            nm = str(nm).strip()
            if nm and nm not in ordered:
                ordered.append(nm)

        reg["names"] = ordered
        reg["specs"] = [seen[nm] for nm in ordered]

        reg["enabled"] = bool(enable)
        if run_during_warmup is not None:
            reg["run_during_warmup"] = bool(run_during_warmup)

        self._ensure_hook_registered(hook_key)
        return list(reg["names"])

    def list_handlers(self, hook) -> list[str]:
        """
        Return the **ordered list of method names** registered for a given
        EnergyPlus runtime hook.

        Parameters
        ----------
        hook : str | callable
            The hook to inspect. Accepts the same identifiers as
            `register_handlers`:
            - Alias: "begin", "before_hvac", "inside_iter", "after_hvac",
            "after_zone", "after_warmup", "after_get_input"
            - Full runtime callback attribute name (str), e.g.
            "callback_end_system_timestep_after_hvac_reporting"
            - The registration callable itself, e.g.
            `runtime.callback_begin_system_timestep_before_predictor`

        Returns
        -------
        list[str]
            The handlers registered **for that hook**, in dispatch order.

        Examples
        --------
        >>> util.list_handlers("begin")
        ['occupancy_handler', 'co2_set_outdoor_ppm']

        >>> util.list_handlers("callback_end_system_timestep_after_hvac_reporting")
        ['probe_zone_air_and_supply_with_kf']

        >>> util.list_handlers(util.runtime.callback_inside_system_iteration_loop)
        ['zone_pid_controller']
        """
        hook_key, reg = self._get_or_init_hook_registry(hook)
        return list(reg["names"])

    def unregister_handlers(self, hook, names: list[str]) -> list[str]:
        """
        Unregister (by name) one or more handlers for a given EnergyPlus runtime hook.

        This removes matching method names from the internal registry for `hook`
        but does **not** detach the underlying EnergyPlus callback itself; any
        remaining handlers for that hook will still be dispatched.

        Parameters
        ----------
        hook : str | callable
            The hook to modify. Accepts the same identifiers as `register_handlers`:
            - Alias: "begin", "before_hvac", "inside_iter", "after_hvac",
            "after_zone", "after_warmup", "after_get_input"
            - Full runtime attribute name (str), e.g.
            "callback_end_system_timestep_after_hvac_reporting"
            - The registration callable itself, e.g.
            `runtime.callback_begin_system_timestep_before_predictor`
        names : list[str]
            Method names to remove (e.g., ["co2_set_outdoor_ppm", "occupancy_handler"]).
            Names that are not currently registered are ignored.

        Returns
        -------
        list[str]
            The **remaining handlers** registered for this hook, in dispatch order.

        Notes
        -----
        - Matching is done by **method name** (string equality).
        - If a handler name appeared multiple times (should not under normal use),
        **all occurrences** will be removed.
        - Passing an empty list leaves the registry unchanged.

        Examples
        --------
        Remove a single handler from the "begin" (before predictor) hook:

        >>> util.unregister_handlers("begin", ["co2_set_outdoor_ppm"])
        ['occupancy_handler', 'probe_zone_air_and_supply_with_kf']

        Remove two handlers using the explicit runtime attribute name:

        >>> util.unregister_handlers(
        ...     "callback_end_system_timestep_after_hvac_reporting",
        ...     ["probe_zone_air_and_supply_with_kf", "my_logger"]
        ... )
        []

        Use the registration callable directly:

        >>> util.unregister_handlers(util.runtime.callback_inside_system_iteration_loop,
        ...                          ["zone_pid_controller"])
        []
        """
        hook_key, reg = self._get_or_init_hook_registry(hook)
        keep = [(nm, spec) for nm, spec in zip(reg["names"], reg["specs"]) if nm not in set(names)]
        reg["names"] = [nm for nm, _ in keep]
        reg["specs"] = [sp for _, sp in keep]
        return list(reg["names"])

    def enable_hook(self, hook):
        """
        Enable dispatch for a specific EnergyPlus runtime hook without altering the
        registered handlers.

        This flips the per-hook "enabled" flag to **True** so that, on the next
        simulation step where EnergyPlus invokes that hook, the currently
        registered handlers (if any) will be dispatched. It does **not** add,
        remove, or re-order handlers, and it does **not** attach the underlying
        EnergyPlus callback if it was never registered.

        Parameters
        ----------
        hook : str | callable
            Identifies the hook to enable. Accepts the same forms as `register_handlers`:
            - Alias string: "begin", "before_hvac", "inside_iter", "after_hvac",
            "after_zone", "after_warmup", "after_get_input"
            - Full runtime attribute name (str), e.g.
            "callback_begin_system_timestep_before_predictor"
            - The registration callable itself, e.g.
            `runtime.callback_begin_system_timestep_before_predictor`

        Notes
        -----
        - If the hook’s EnergyPlus callback has **never** been attached to the
        current `state`, enabling here won’t attach it. Use `register_handlers(...)`
        at least once for that hook to ensure the callback is bound.
        - This does not change the hook’s **warmup** policy. If the hook is set to
        skip warmup (default), enabling it will still skip dispatch during warmup.

        Returns
        -------
        None

        Examples
        --------
        Enable the “before predictor” (begin system timestep) hook by alias:

        >>> util.enable_hook("begin")

        Enable by passing the runtime registration function:

        >>> util.enable_hook(util.runtime.callback_begin_system_timestep_before_predictor)
        """
        _, reg = self._get_or_init_hook_registry(hook)
        reg["enabled"] = True

    def disable_hook(self, hook):
        """
        Disable dispatch for a specific EnergyPlus runtime hook without changing
        which handlers are registered.

        This flips the per-hook "enabled" flag to **False** so that, even if the
        underlying EnergyPlus callback fires, the utility’s dispatcher will **skip**
        invoking your handlers for that hook. It does **not** add, remove, or
        re-order handlers, and it does **not** detach the underlying EnergyPlus
        callback from the current state.

        Parameters
        ----------
        hook : str | callable
            Identifies the hook to disable. Accepts the same forms as `register_handlers`:
            - Alias string: "begin", "before_hvac", "inside_iter", "after_hvac",
            "after_zone", "after_warmup", "after_get_input"
            - Full runtime attribute name (string), e.g.
            "callback_begin_system_timestep_before_predictor"
            - The registration callable itself, e.g.
            `runtime.callback_inside_system_iteration_loop`

        Notes
        -----
        - Disabling a hook keeps its handlers in the registry. Use
        `unregister_handlers(hook, [...])` or `register_handlers(hook, [], clear=True)`
        if you want to actually remove them.
        - Disabling does not affect the hook’s **warmup** policy. If you need to change
        whether handlers run during warmup, call `register_handlers(...)` with the
        `run_during_warmup` argument.
        - To re-enable dispatch, call `enable_hook(hook)`.

        Returns
        -------
        None

        Examples
        --------
        Temporarily pause “begin system timestep (before predictor)” dispatch:

        >>> util.disable_hook("begin")
        >>> util.run_design_day()  # handlers for "begin" will not run
        >>> util.enable_hook("begin")  # resume dispatch later

        Disable by passing the runtime registration function:

        >>> util.disable_hook(util.runtime.callback_inside_system_iteration_loop)
        """
        _, reg = self._get_or_init_hook_registry(hook)
        reg["enabled"] = False

    # --- back-compat thin shims (optional) ---

    # def register_begin_iteration(self, methods, **kw):
    #     """Back-compat wrapper: runs at the beginning of each system timestep (before predictor)."""
    #     return self.register_handlers("begin", methods, **kw)

    # def register_after_hvac_reporting(self, methods, **kw):
    #     """Back-compat wrapper: runs after HVAC reporting each system timestep."""
    #     return self.register_handlers("after_hvac", methods, **kw)    

    def enable_runtime_logging(self):
        def _on_msg(msg):
            s = msg.decode("utf-8", errors="ignore") if isinstance(msg, (bytes, bytearray)) else str(msg)
            print(s.rstrip())
        self._runtime_log_enabled = True
        self._runtime_log_func = _on_msg
        self.runtime.callback_message(self.state, _on_msg)

    def disable_runtime_logging(self):
        self._runtime_log_enabled = False
        self._runtime_log_func = None
        # no unregister API; future resets will simply not re-register

    def override_hvac_schedules(
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
                    for nm in (self.exchange.get_object_names(s, typ) or []):
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
                        h = self.exchange.get_actuator_handle(s, typ, "Schedule Value", nm)
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
            if not getattr(self, "_hvac_kill_enabled", False) or self.exchange.warmup_flag(s):
                return
            for h in getattr(self, "_hvac_kill_handles", []):
                # Force schedule to zero each timestep
                self.exchange.set_actuator_value(s, h, 0.0)

        # Register with your unified registrar (so runs survive reset_state)
        pair_warmup = (self.runtime.callback_after_new_environment_warmup_complete, _after_warmup)
        pair_tick   = (self.runtime.callback_begin_system_timestep_before_predictor, _on_tick)

        # avoid duplicate registrations if called twice
        for pair in (pair_warmup, pair_tick):
            if pair not in self._extra_callbacks:
                self._extra_callbacks.append(pair)

        # if someone calls this *after* set_model but *before* a run, make sure callbacks are queued
        # (run_design_day/run_annual call _register_callbacks() after reset_state, so no need to call it here)

    def release_hvac_schedules(self) -> None:
        self._hvac_kill_enabled = False
        self._hvac_kill_handles = []

    def probe_zone_air_and_supply(self, s, **opts):
        """
        Fast per-timestep callback that snapshots **zone air state** and **supply (inlet) aggregates**,
        with robust fallbacks and optional logging. Designed to be registered via
        `register_begin_iteration(...)` or `register_after_hvac_reporting(...)`, but it can also
        be called directly.

        What it does
        ------------
        • Discovers each zone’s inlet node(s) (prefers SQL; falls back to common name patterns).  
        • Requests the minimal set of E+ variables once per state and then resolves handles when
        `api_data_fully_ready(...)` becomes True.  
        • At each call (post-warmup), builds a snapshot containing:
            - **Outdoor**: dry-bulb (°C), humidity ratio (kg/kg), CO₂ (ppm).
            - **Per zone**:
                - **air**: dry-bulb (°C), humidity ratio (kg/kg), CO₂ (ppm).
                - **supply** (inlet aggregate): mass flow (kg/s), dry-bulb (°C),
                humidity ratio (kg/kg), CO₂ (ppm) — mass-flow-weighted when multiple nodes exist.
            - **inlet_nodes**: the node names used for that zone.

        Humidity-ratio fallbacks (w)
        ----------------------------
        If a direct humidity-ratio variable is unavailable, the function computes **w** via a
        Tetens-based relation using (T[°C], RH[%], P[Pa]):

        • **Zone w**: try *Zone Air Humidity Ratio* → *Zone Mean Air Humidity Ratio* → compute from zone T, zone RH, site P.  
        • **Site w**: try *Site Outdoor Air Humidity Ratio* → compute from site T, site RH, site P.  
        • **Supply node w**: try *System Node Humidity Ratio* → compute from node T, node RH, site P.

        CO₂ fallback
        ------------
        If *Site Outdoor Air CO2 Concentration* is absent and a CO₂ outdoor schedule actuator
        handle (`self._co2_outdoor_sched_handle`) is available, that value is used instead.

        Parameters
        ----------
        s : EnergyPlusState
            The active EnergyPlus runtime state passed by the callback system.
        **opts :
            Optional keyword arguments:
            • `log_every_minutes: int | None = 1`
                - `1` → print once each model minute (i.e., every timestep if 60/N = 1).
                - `None` → disable prints.
                Printing is gated so each timestamp is logged at most once.
            • `precision: int = 3`
                Decimal precision for printed values.

        Returns
        -------
        dict | None
            A snapshot dict (also stored on `self._probe_last_snapshot`) of the form:

            {
            "timestamp": pandas.Timestamp | str,
            "outdoor": {"Tdb_C": float, "w_kgperkg": float, "co2_ppm": float},
            "zones": {
                "<ZONE>": {
                "air":    {"Tdb_C": float, "w_kgperkg": float, "co2_ppm": float},
                "supply": {"m_dot_kgs": float, "Tdb_C": float, "w_kgperkg": float, "co2_ppm": float},
                "inlet_nodes": [str, ...]
                },
                ...
            }
            }

            Returns `None` during warmup or before variable handles are ready.

        Side effects
        ------------
        • Issues `request_variable(...)` calls once, resolves and caches handles once per state,
        and writes concise log lines via `self._log(...)` when `log_every_minutes` is not None.  
        • Saves the latest payload to `self._probe_last_snapshot`.

        Notes
        -----
        • Missing/unknown readings are `numpy.nan`.  
        • Supply aggregates use **mass-flow weighting** over all discovered inlet nodes.  
        • Handles used here are valid only for the current temporary state; do not reuse them elsewhere.

        Example
        -------
        Register to print once every 15 minutes (model time) with 2-decimal precision:

        >>> util.register_begin_iteration([
        ...   {"method_name": "probe_zone_air_and_supply",
        ...    "kwargs": {"log_every_minutes": 15, "precision": 2}}
        ... ])
        >>> util.run_design_day()  # or util.run_annual()

        Or call directly inside a custom callback:

        >>> snap = util.probe_zone_air_and_supply(util.state, log_every_minutes=None)
        >>> snap["zones"]["LIVING"]["air"]["Tdb_C"]
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

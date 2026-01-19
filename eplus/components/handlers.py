from typing import List, Tuple, Callable

class HandlersMixin:
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
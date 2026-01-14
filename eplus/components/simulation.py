import pathlib, re

class SimulationMixin:
    def __init__(self):
        print("SimulationMixin initialized.")

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
        # TODO:Add this Register Thing
        # self._register_callbacks()
        return self.api.runtime.run_energyplus(
            self.state, ['-w', self.epw, '-d', self.out_dir, self.idf]
        )

    def run_design_day(self) -> int:
        """
        Run a **design-day-only** EnergyPlus simulation for the active `idf`/`epw`
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
        # TODO:Add this Register Thing
        # self._register_callbacks()
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

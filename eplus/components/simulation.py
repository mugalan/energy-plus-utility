import pathlib, re
from typing import List, Dict, Tuple, Optional, Sequence

class SimulationMixin:
    def __init__(self):
        self._log(2, "Initialized SimulationMixin")

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
        return self.runtime.run_energyplus(
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
        self._register_callbacks()
        return self.runtime.run_energyplus(
            self.state, ['-w', self.epw, '-d', self.out_dir, '--design-day', self.idf]
        )

    def run_dry_run(self, *, include_ems_edd: bool = False, reset: bool = True, design_day: bool = True) -> int:
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
        return self.runtime.run_energyplus(self.state, args)

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
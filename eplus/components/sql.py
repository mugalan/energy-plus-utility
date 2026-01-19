import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
from typing import List, Dict, Tuple, Optional, Sequence

class SQLMixin:
    def __init__(self):
        self._log(2, "Initialized SQLMixin")

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
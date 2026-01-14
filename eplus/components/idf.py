import os
import shutil
from typing import Optional
import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib
import pandas as pd

class IDFMixin:
    def __init__(self):
        self._log(2, "Initialized IDFMixin")
        self._patched_idf_path: Optional[str] = None
        self._orig_idf_path: Optional[str] = None
    
    # TODO:Add to ColabDOCs
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

    def api_catalog_df(self, *, save_csv: bool = False) -> dict[str, "pd.DataFrame"]:
        """
        Discover **runtime API-exposed catalogs** from EnergyPlus and return them as
        pandas DataFrames, grouped by section.

        Under the hood this wraps:
            self.api.exchange.list_available_api_data_csv(self.state)

        What you get
        ------------
        A dict mapping **section name → DataFrame**, for *all* sections present in
        the current model / E+ build. Typical keys you may see:
        - "ACTUATORS"
        - "INTERNAL_VARIABLES"
        - "PLUGIN_GLOBAL_VARIABLES"
        - "TRENDS"
        - "METERS"
        - "VARIABLES"

        Notes & scope
        -------------
        • This catalog comes **directly from the runtime API** (no IDF parsing, no RDD/MDD/EDD).
        • Availability depends on when you call it; best after inputs are parsed or API data are ready.
        Use one of:
            - inside `callback_after_get_input`, or
            - after warmup via `callback_after_new_environment_warmup_complete`, or
            - when `self.api.exchange.api_data_fully_ready(self.state)` is True.
        • Column shapes vary slightly across sections / versions. This function assigns
        sensible headers per known section and pads/truncates rows as needed.

        Parameters
        ----------
        save_csv : bool, default False
            If True, writes the **raw** CSV from EnergyPlus to `<out_dir>/api_catalog.csv`.

        Returns
        -------
        dict[str, pandas.DataFrame]
            A dictionary of DataFrames keyed by section name. Missing sections simply won't appear.

        Examples
        --------
        >>> # Get everything the runtime reports
        >>> sections = util.api_catalog_df()
        >>> list(sections.keys())
        ['ACTUATORS', 'INTERNAL_VARIABLES', 'PLUGIN_GLOBAL_VARIABLES', 'TRENDS', 'METERS', 'VARIABLES']

        >>> # Inspect schedule-based actuators you can set via get_actuator_handle(...)
        >>> acts = sections.get("ACTUATORS", pd.DataFrame())
        >>> acts.query("ComponentType == 'Schedule:Compact' and ControlType == 'Schedule Value'").head()

        >>> # See available report variables (names/keys/units) the API knows about
        >>> vars_df = sections.get("VARIABLES", pd.DataFrame())
        >>> vars_df.head()

        >>> # Save the raw catalog for auditing
        >>> util.api_catalog_df(save_csv=True)
        """

        ex = self.exchange
        csv_bytes = ex.list_available_api_data_csv(self.state)
        

        # Optionally persist the raw CSV
        if save_csv:
            try:
                out_path = os.path.join(self.out_dir, "api_catalog.csv")
                with open(out_path, "wb") as f:
                    f.write(csv_bytes)
                try:
                    self._log(1, f"[api_catalog] Saved → {out_path} ({len(csv_bytes)} bytes)")
                except Exception:
                    print(f"[api_catalog] Saved → {out_path} ({len(csv_bytes)} bytes)")
            except Exception:
                pass

        # Parse the catalog: the file is a sequence of sections, each starting with "**NAME**"
        lines = csv_bytes.decode("utf-8", errors="replace").splitlines()
        sections_raw: dict[str, list[list[str]]] = {}
        current = None
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("**") and line.endswith("**"):
                current = line.strip("*").strip().upper().replace(" ", "_")
                sections_raw.setdefault(current, [])
                continue
            # Catalog rows are simple CSV without quoted commas → split on ','
            row = [c.strip() for c in line.split(",")]
            if current:
                sections_raw[current].append(row)

        # Known schemas per section (fallbacks are applied when row lengths differ)
        SCHEMAS: dict[str, list[str]] = {
            # Example row: Actuator,Schedule:Compact,Schedule Value,OCCUPY-1,[ ]
            "ACTUATORS": ["Kind", "ComponentType", "ControlType", "ActuatorKey", "Units"],
            # Example row: Internal Variable,Zone,Zone Floor Area,LIVING ZONE,[m2]
            "INTERNAL_VARIABLES": ["Kind", "VariableType", "VariableName", "KeyValue", "Units"],
            # Example row: Plugin Global Variable,<name>
            "PLUGIN_GLOBAL_VARIABLES": ["Kind", "Name"],
            # Example row: Trend,<name>,<length> (varies)
            "TRENDS": ["Kind", "Name", "Length"],
            # Example row: Meter,Electricity:Facility,[J] (varies)
            "METERS": ["Kind", "MeterName", "Units"],
            # Example row: Variable,Zone Mean Air Temperature,LIVING ZONE,[C] (varies)
            "VARIABLES": ["Kind", "VariableName", "KeyValue", "Units"],
        }

        dfs: dict[str, pd.DataFrame] = {}
        for sec, rows in sections_raw.items():
            # Choose schema or a generic fallback wide enough for the observed rows
            cols = SCHEMAS.get(sec)
            if cols is None:
                max_cols = max([len(r) for r in rows] + [5])
                cols = [f"col{i+1}" for i in range(max_cols)]

            # Normalize rows to the column count
            width = len(cols)
            norm = [(r + [""] * (width - len(r)))[:width] for r in rows]
            df = pd.DataFrame(norm, columns=cols)

            # Light cleanup
            if "Kind" in df.columns:
                df["Kind"] = df["Kind"].astype(str).str.strip().str.title()
            for c in df.columns:
                df[c] = df[c].astype(str).str.strip()

            dfs[sec] = df

        return dfs

    def list_available_variables(self, *, save_csv: bool = False):
        """
        Return the **runtime API catalog of report variables** as a pandas DataFrame.

        What this is
        ------------
        A thin wrapper around `self.api_catalog_df()` that extracts the "VARIABLES"
        section reported by the EnergyPlus runtime API (via
        `exchange.list_available_api_data_csv`). It does **not** parse your IDF and
        does **not** require RDD/MDD/SQL — it’s whatever the API exposes at runtime.

        When to call
        ------------
        Call after inputs are parsed (e.g., in/after `callback_after_get_input`) or
        once `exchange.api_data_fully_ready(self.state)` is True. Calling earlier may
        yield an empty frame.

        Columns (typical)
        -----------------
        ["Kind", "VariableName", "KeyValue", "Units"]
        (Column names are normalized by `api_catalog_df`; may vary slightly by E+ version.)

        Parameters
        ----------
        save_csv : bool, default False
            If True, also saves the **raw** API catalog CSV to `<out_dir>/api_catalog.csv`.

        Returns
        -------
        pandas.DataFrame
            The "VARIABLES" section; empty DataFrame if the section is absent.

        Examples
        --------
        >>> df = util.list_available_variables()
        >>> df.head()

        >>> # What zone-style variables are available?
        >>> df[df["VariableName"].str.contains("Zone ", case=False, na=False)].head()
        """
        import pandas as pd
        sections = self.api_catalog_df(save_csv=save_csv)
        df = sections.get("VARIABLES", pd.DataFrame(columns=["Kind","VariableName","KeyValue","Units"]))
        return df

    def list_available_meters(self, *, save_csv: bool = False):
        """
        Return the **runtime API catalog of meters** as a pandas DataFrame.

        What this is
        ------------
        A convenience accessor for the "METERS" section from
        `exchange.list_available_api_data_csv`. Unlike RDD/MDD parsing, this is
        **purely runtime** — no dependency on dictionary files.

        When to call
        ------------
        After API data are available (post input parsing / warmup). Earlier calls may
        return an empty frame depending on the model & E+ version.

        Columns (typical)
        -----------------
        ["Kind", "MeterName", "Units"]

        Parameters
        ----------
        save_csv : bool, default False
            If True, also saves the raw API catalog CSV to `<out_dir>/api_catalog.csv`.

        Returns
        -------
        pandas.DataFrame
            The "METERS" section; empty DataFrame if the section is absent.

        Examples
        --------
        >>> meters = util.list_available_meters()
        >>> meters.query("MeterName.str.contains('Electricity', case=False)", engine='python').head()
        """
        import pandas as pd
        sections = self.api_catalog_df(save_csv=save_csv)
        df = sections.get("METERS", pd.DataFrame(columns=["Kind","MeterName","Units"]))
        return df

    def list_available_actuators(self, *, save_csv: bool = False):
        """
        Return the **runtime API catalog of actuators** as a pandas DataFrame.

        What this is
        ------------
        A small wrapper that extracts the "ACTUATORS" section from the runtime API
        catalog (`exchange.list_available_api_data_csv`). Use these rows to look up
        actuator **handles** during a run.

        When to call
        ------------
        After inputs are parsed / API data are ready (e.g., inside
        `callback_after_component_get_input` or after warmup). Earlier calls can be empty.

        Columns (typical)
        -----------------
        ["Kind", "ComponentType", "ControlType", "ActuatorKey", "Units"]

        Getting handles
        ---------------
        At an appropriate callback (when data are ready), resolve a handle with:
            `h = ex.get_actuator_handle(state, ComponentType, ControlType, ActuatorKey)`
        Then set values each timestep via:
            `ex.set_actuator_value(state, h, value)`

        Parameters
        ----------
        save_csv : bool, default False
            If True, also saves the raw API catalog CSV to `<out_dir>/api_catalog.csv`.

        Returns
        -------
        pandas.DataFrame
            The "ACTUATORS" section; empty DataFrame if the section is absent.

        Examples
        --------
        >>> acts = util.list_available_actuators()
        >>> # All schedule knobs you can drive
        >>> acts.query("ComponentType == 'Schedule:Compact' and ControlType == 'Schedule Value'").head()

        >>> # Example: find a specific fan/coil actuator family
        >>> acts[acts["ComponentType"].str.contains("Fan|Coil", case=False, na=False)].head()
        """
        import pandas as pd
        sections = self.api_catalog_df(save_csv=save_csv)
        df = sections.get("ACTUATORS", pd.DataFrame(columns=["Kind","ComponentType","ControlType","ActuatorKey","Units"]))
        return df

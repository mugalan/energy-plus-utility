import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib
from typing import List, Dict, Tuple, Optional, Sequence
import sqlite3

class UtilsMixin:
    def __init__(self):
        self._log(2, "Initialized UtilsMixin")
        
    def _assert_out_dir_writable(self):
        assert self.out_dir, "set_model(...) first."
        os.makedirs(self.out_dir, exist_ok=True)
        # quick write test
        tmp = pathlib.Path(self.out_dir) / ".write_test.tmp"
        with open(tmp, "wb") as f:
            f.write(b"ok")
        tmp.unlink(missing_ok=True)

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
        isn't present.
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
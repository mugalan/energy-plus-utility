from __future__ import annotations

import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

import pandas as pd


class EPlusSqlExplorer:
    """
    Small helper around EnergyPlus's eplusout.sql.

    Example:
        xp = EPlusSqlExplorer("eplus_out/eplusout.sql")
        xp.list_tables()
        xp.peek("ReportData", 5)
        hits = xp.search_value("Electricity:Facility")
        df = xp.auto_extract_series("Electricity:Facility", to_kwh=True)
    """

    def __init__(self, sql_path: str = os.path.join("eplus_out", "eplusout.sql"), *, verbose: bool = False):
        self.sql_path = sql_path
        self.verbose = bool(verbose)

    # ---------- internals ----------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _open_conn(self) -> sqlite3.Connection:
        if not os.path.exists(self.sql_path):
            raise FileNotFoundError(self.sql_path)
        return sqlite3.connect(self.sql_path)

    # ---------- quick views ----------
    def list_tables(self) -> List[Tuple[str, Optional[int]]]:
        """Return list of (table_name, row_count)."""
        conn = self._open_conn()
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )]
            out: List[Tuple[str, Optional[int]]] = []
            for t in tables:
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM \"{t}\"").fetchone()[0]
                except Exception:
                    n = None
                out.append((t, n))
            return out
        finally:
            conn.close()

    def table_schema(self, table: str) -> List[Tuple]:
        """Return PRAGMA table_info for a table: (cid, name, type, notnull, dflt_value, pk)."""
        conn = self._open_conn()
        try:
            return conn.execute(f"PRAGMA table_info(\"{table}\")").fetchall()
        finally:
            conn.close()

    def peek(self, table: str, limit: int = 10) -> pd.DataFrame:
        """Return first rows of a table as a DataFrame."""
        conn = self._open_conn()
        try:
            return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(limit)}', conn)
        finally:
            conn.close()

    # ---------- helpers ----------
    @staticmethod
    def _find_text_columns(conn: sqlite3.Connection, table: str) -> List[str]:
        cols = conn.execute(f"PRAGMA table_info(\"{table}\")").fetchall()
        textish: List[str] = []
        for (_cid, name, ctype, *_rest) in cols:
            t = (ctype or "").upper()
            if any(k in t for k in ("CHAR", "TEXT", "CLOB")) or t == "" or "NVARCHAR" in t:
                textish.append(name)
        return textish

    @staticmethod
    def _detect_minute_col(conn: sqlite3.Connection) -> str:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(Time)").fetchall()}
        if "Minute" in cols: return "Minute"
        if "Minutes" in cols: return "Minutes"
        return "Minute"

    # ---------- search ----------
    def search_value(
        self,
        value: str,
        *,
        tables: Optional[Sequence[str]] = None,
        limit_per_table: int = 5
    ) -> Dict[str, List[Tuple[str, List[int]]]]:
        """
        Search a literal value across TEXT-like columns of all (or provided) tables.
        Returns: {table: [(column, [rowids...]), ...], ...}
        """
        conn = self._open_conn()
        try:
            all_tables = list(tables) if tables else [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            hits: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)
            for t in all_tables:
                text_cols = self._find_text_columns(conn, t)
                for c in text_cols:
                    q = f'SELECT rowid FROM "{t}" WHERE "{c}" = ? LIMIT ?'
                    try:
                        rows = conn.execute(q, (value, int(limit_per_table))).fetchall()
                    except Exception:
                        continue
                    if rows:
                        hits[t].append((c, [r[0] for r in rows]))
            return hits
        finally:
            conn.close()

    # ---------- series extraction ----------
    def auto_extract_series(
        self,
        value: str = "Electricity:Facility",
        *,
        freq_whitelist: Sequence[str] = ("TimeStep", "Hourly"),
        include_design_days: bool = False,
        to_kwh: bool = True,
        csv_out: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Locate `value` (e.g., a meter/variable label) in the schema, find its paired data table,
        join to Time (and EnvironmentPeriods), and return a tidy DataFrame ['timestamp','value'].

        - Converts J→kWh if `to_kwh=True`.
        - If `csv_out` is provided, writes a wide CSV with an auto-named column.
        - Returns None if no rows found.
        """
        conn = self._open_conn()
        try:
            # 1) find candidate dictionary table/column
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            hits = self.search_value(value, tables=tables, limit_per_table=1)
            if not hits:
                self._log(f"'{value}' not found in any text column.")
                return None

            dict_table = next(iter(hits.keys()))
            dict_col   = hits[dict_table][0][0]

            # dictionary index column
            dict_info = conn.execute(f'PRAGMA table_info("{dict_table}")').fetchall()
            idx_cols = [name for (_cid, name, _type, _nn, _df, _pk) in dict_info if name.endswith("Index")]
            if not idx_cols:
                idx_cols = [
                    name for (_cid, name, ctype, _n, _d, pk) in dict_info
                    if pk == 1 and "INT" in (ctype or "").upper()
                ]
            if not idx_cols:
                raise RuntimeError(f"No index column found in {dict_table} (looking for '*Index').")
            dict_idx_col = idx_cols[0]

            # dictionary index value for this label (if possible)
            try:
                res = conn.execute(
                    f'SELECT "{dict_idx_col}" FROM "{dict_table}" WHERE "{dict_col}" = ?',
                    (value,)
                ).fetchone()
                dict_idx_val = res[0] if res else None
            except Exception:
                dict_idx_val = None

            # 2) find a data table that references this dictionary and Time
            data_table = None
            for t in tables:
                cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()}
                if dict_idx_col in cols and "TimeIndex" in cols:
                    data_table = t
                    break
            if not data_table:
                raise RuntimeError(f"No data table found referencing '{dict_idx_col}' with TimeIndex.")

            # numeric value column
            data_info = conn.execute(f'PRAGMA table_info("{data_table}")').fetchall()
            val_col = "Value" if any(n == "Value" for (_cid, n, _t, *_rest) in data_info) else None
            if not val_col:
                for (_cid, n, t, *_rest) in data_info:
                    tt = (t or "").upper()
                    if any(k in tt for k in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")) and n not in ("TimeIndex", dict_idx_col):
                        val_col = n
                        break
            if not val_col:
                raise RuntimeError(f"No numeric value column found in {data_table}.")

            # frequency column (optional)
            freq_col = None
            dict_cols = {r[1] for r in dict_info}
            for cand in ("ReportingFrequency", "ReportingInterval", "Interval", "Frequency"):
                if cand in dict_cols:
                    freq_col = cand
                    break

            # environment filter (exclude sizing periods unless requested)
            env_filter = ""
            env_params: List = []
            env_cols = {r[1] for r in conn.execute('PRAGMA table_info("EnvironmentPeriods")').fetchall()}
            if not include_design_days:
                if "EnvironmentType" in env_cols:
                    env_filter = "AND ep.EnvironmentType = ?"
                    env_params = ["WeatherRunPeriod"]
                elif "EnvironmentName" in env_cols:
                    env_filter = "AND ep.EnvironmentName NOT LIKE 'SizingPeriod:%'"

            minute_col = self._detect_minute_col(conn)

            # Build WHERE clauses
            if dict_idx_val is not None:
                base_where = f'WHERE d."{dict_col}" = ? AND x."{dict_idx_col}" = ?'
                base_params: List = [value, dict_idx_val]
            else:
                base_where = f'WHERE d."{dict_col}" = ? AND x."{dict_idx_col}" = d."{dict_idx_col}"'
                base_params = [value]

            freq_clause = ""
            freq_params: List = []
            if freq_col and freq_whitelist:
                ph = ",".join("?" * len(freq_whitelist))
                freq_clause = f'AND d."{freq_col}" IN ({ph})'
                freq_params = list(freq_whitelist)

            sql = f"""
                SELECT
                  d."{dict_col}" AS name,
                  t.Year AS y, t.Month AS m, t.Day AS d, t.Hour AS h, t."{minute_col}" AS mi,
                  x."{val_col}" AS val
                FROM "{data_table}" x
                JOIN "{dict_table}" d ON x."{dict_idx_col}" = d."{dict_idx_col}"
                JOIN "Time" t ON x.TimeIndex = t.TimeIndex
                LEFT JOIN "EnvironmentPeriods" ep ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
                {base_where}
                {freq_clause}
                {env_filter}
            """
            params = base_params + freq_params + env_params
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                self._log("Query returned no rows; adjust frequency or include_design_days.")
                return None

            df = pd.DataFrame(rows, columns=["name", "y", "m", "d", "h", "min", "value"])

            # timestamp (E+ hour is end-of-interval → shift to start; Year 0 → 2002)
            y = df["y"].replace(0, 2002)
            df["timestamp"] = pd.to_datetime(
                dict(year=y, month=df["m"], day=df["d"], hour=(df["h"] - 1).clip(lower=0), minute=df["min"]),
                errors="coerce"
            )
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")[["timestamp", "value"]]

            if to_kwh:
                df["value"] = df["value"] / 3.6e6  # J → kWh

            if csv_out:
                col = value.replace(":", "_") + ("_kWh" if to_kwh else "_J")
                out = df.rename(columns={"value": col})
                out.to_csv(csv_out, index=False)
                self._log(f"Wrote CSV → {csv_out}  shape={out.shape}")

            return df
        finally:
            conn.close()

    def list_sql_variables(
        self,
        *,
        name: str | None = None,          # exact variable name (e.g., "System Node Mass Flow Rate")
        name_like: str | None = None,     # SQL LIKE pattern (e.g., "System Node %")
        key: str | None = None,           # exact KeyValue (e.g., "SPACE1-1 In Node")
        key_like: str | None = None,      # SQL LIKE pattern for KeyValue (e.g., "%In Node%")
        reporting_freq: tuple[str, ...] | None = None,  # e.g., ("TimeStep","Zone Timestep")
        include_design_days: bool = False,
        is_meter: bool | None = False,    # None → both; False → variables only; True → meters only
        limit: int | None = None
    ) -> pd.DataFrame:
        """
        Return a DataFrame of what’s actually in eplusout.sql (from ReportData/ReportDataDictionary).

        Columns: [Name, KeyValue, Units, ReportingFrequency, IsMeter, n_rows]

        Examples:
        # All system-node entries (any variable under "System Node ...")
        df = util.list_sql_variables(name_like="System Node %")

        # Where do we have "System Node Mass Flow Rate" and for which nodes?
        df = util.list_sql_variables(name="System Node Mass Flow Rate")

        # Narrow to node keys that look like SPACE1-1 "In Node"
        df = util.list_sql_variables(name="System Node Mass Flow Rate", key_like="%SPACE1-1%In Node%")

        # See all Zone variables that have rows (useful sanity check)
        df = util.list_sql_variables(name_like="Zone %")

        # Include sizing/design days too
        df = util.list_sql_variables(name_like="System Node %", include_design_days=True)
        """
        sql_path = self.sql_path
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"{sql_path} not found. Run a simulation with Output:SQLite enabled.")

        # Build WHERE clauses
        where = ["1=1"]
        params: list = []

        if is_meter is True:
            where.append("(d.IsMeter = 1)")
        elif is_meter is False:
            where.append("(d.IsMeter = 0 OR d.IsMeter IS NULL)")

        if name is not None:
            where.append("d.Name = ?")
            params.append(name)
        elif name_like is not None:
            where.append("UPPER(d.Name) LIKE UPPER(?)")
            params.append(name_like)

        if key is not None:
            where.append("(d.KeyValue = ?)")
            params.append(key)
        elif key_like is not None:
            where.append("(UPPER(COALESCE(d.KeyValue,'')) LIKE UPPER(?))")
            params.append(key_like)

        if reporting_freq:
            # allow tolerant matches (LIKE %HOUR% etc.) similar to your helpers
            rf_map = {
                "TIMESTEP": ["%TIMESTEP%", "DETAILED"],
                "HOURLY":   ["%HOUR%"],
                "DAILY":    ["%DAY%"],
                "MONTHLY":  ["%MONTH%"],
                "RUNPERIOD":["%RUN%PERIOD%"],
                "ZONE TIMESTEP": ["%ZONE%TIMESTEP%"],
                "SYSTEM TIMESTEP": ["%SYSTEM%TIMESTEP%"],
            }
            sub_ors = []
            for f in reporting_freq:
                fkey = str(f).upper()
                pats = rf_map.get(fkey, [fkey])
                terms = []
                for p in pats:
                    if p.startswith("%"):
                        terms.append("UPPER(d.ReportingFrequency) LIKE ?")
                        params.append(p)
                    else:
                        terms.append("UPPER(d.ReportingFrequency) = ?")
                        params.append(p)
                sub_ors.append("(" + " OR ".join(terms) + ")")
            if sub_ors:
                where.append("(" + " OR ".join(sub_ors) + ")")

        env_clause = "" if include_design_days else \
            "AND (ep.EnvironmentName IS NULL OR ep.EnvironmentName NOT LIKE 'SizingPeriod:%')"

        q = f"""
            SELECT
                d.Name,
                COALESCE(d.KeyValue,'') AS KeyValue,
                COALESCE(d.Units,'') AS Units,
                COALESCE(d.ReportingFrequency,'') AS ReportingFrequency,
                COALESCE(d.IsMeter, 0) AS IsMeter,
                COUNT(*) AS n_rows
            FROM ReportData r
            JOIN ReportDataDictionary d
                ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
            JOIN Time t ON r.TimeIndex = t.TimeIndex
            LEFT JOIN EnvironmentPeriods ep
                ON t.EnvironmentPeriodIndex = ep.EnvironmentPeriodIndex
            WHERE {" AND ".join(where)}
            {env_clause}
            GROUP BY d.Name, d.KeyValue, d.Units, d.ReportingFrequency, d.IsMeter
            ORDER BY n_rows DESC, d.Name, d.KeyValue
        """

        conn = sqlite3.connect(sql_path)
        try:
            df = pd.read_sql_query(q, conn, params=params)
            if limit is not None and limit > 0:
                df = df.head(int(limit))
            return df
        finally:
            conn.close()


    def list_sql_distinct_names(
        self,
        *,
        name_like: str | None = None,
        is_meter: bool | None = False
    ) -> pd.DataFrame:
        """
        Quick list of distinct variable (or meter) names that actually have rows in SQL.

        Example:
        util.list_sql_distinct_names(name_like="System Node %")
        """
        assert self.out_dir, "Call set_model(...) first."
        sql_path = os.path.join(self.out_dir, "eplusout.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"{sql_path} not found.")

        where = ["1=1"]
        params: list = []
        if is_meter is True:
            where.append("(d.IsMeter = 1)")
        elif is_meter is False:
            where.append("(d.IsMeter = 0 OR d.IsMeter IS NULL)")
        if name_like:
            where.append("UPPER(d.Name) LIKE UPPER(?)")
            params.append(name_like)

        q = f"""
            SELECT d.Name, COUNT(*) AS n_rows
            FROM ReportData r
            JOIN ReportDataDictionary d
                ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
            WHERE {" AND ".join(where)}
            GROUP BY d.Name
            ORDER BY n_rows DESC, d.Name
        """

        conn = sqlite3.connect(sql_path)
        try:
            return pd.read_sql_query(q, conn, params=params)
        finally:
            conn.close()


    def get_table_data(self, db=None, table="KalmanEstimates",
                        *, timestamp_candidates=("Timestamp","DateTime","dateTime","TIME","Time","ts","time_stamp","date_time"),
                        verbose=True):
        """
        Load *all* rows from a given SQLite table and return them as a DataFrame.

        Generalized behavior:
        - Works with any DB filename and table name (SQLite).
        - Detects a timestamp-like column (prefers 'Timestamp' if present, else tries common candidates, case-insensitive).
        - Parses that column to pandas datetime (errors='coerce').
        - Prints quick stats: row count, time range (if timestamp found), top 'Zone' (case-insensitive) values.

        Returns:
        pandas.DataFrame (possibly with a parsed datetime column).
        """

        # Resolve DB path
        path = db else self.sql_path
        if not os.path.exists(path):
            if verbose:
                print(f"[check] DB not found: {path}")
            return None

        # SQLite identifier quoting
        def _q(ident: str) -> str:
            return '"' + str(ident).replace('"', '""') + '"'

        with sqlite3.connect(path) as con:
            # 1) Table exists?
            tabs = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
            names = set(tabs["name"].astype(str).tolist())
            if table not in names:
                if verbose:
                    print(f"[check] Table '{table}' NOT found in {path}. Available: {sorted(names)}")
                return None

            # 2) Row count (cheap)
            n = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {_q(table)}", con)["n"].iat[0]
            if verbose:
                print(f"[check] {os.path.basename(path)} | table={table} : {n} rows")

            # 3) Pull schema to detect columns
            schema = pd.read_sql_query(f"PRAGMA table_info({_q(table)})", con)
            cols = schema["name"].astype(str).tolist()
            cols_lower = {c.lower(): c for c in cols}

            # Timestamp detection (prefer exact 'Timestamp' if present)
            ts_col = None
            if "timestamp" in cols_lower:
                ts_col = cols_lower["timestamp"]
            else:
                # Try candidates in order, case-insensitive
                for cand in timestamp_candidates:
                    c = cand.lower()
                    if c in cols_lower:
                        ts_col = cols_lower[c]
                        break

            # Zone detection (case-insensitive)
            zone_col = cols_lower.get("zone", None)

            # 4) Load all rows
            df = pd.read_sql_query(f"SELECT * FROM {_q(table)}", con)

        # 5) Parse the timestamp column if found
        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            if verbose:
                # Compute time range safely
                if not df[ts_col].dropna().empty:
                    start, end = df[ts_col].min(), df[ts_col].max()
                    print(f"[check] Time range ({ts_col}): {start} → {end}")
                else:
                    print(f"[check] Timestamp column '{ts_col}' present but could not parse any valid datetimes.")
        else:
            if verbose:
                print(f"[check] No timestamp-like column found. Searched: "
                    f"{['Timestamp'] + list(timestamp_candidates)}")

        # 6) Top zones (if column exists)
        if zone_col and zone_col in df.columns:
            try:
                topz = (df[zone_col].astype(str)
                            .value_counts()
                            .head(10)
                            .reset_index())
                topz.columns = [zone_col, "n"]
                if verbose:
                    print(f"[check] Top zones: {topz.to_dict('records')}")
            except Exception:
                if verbose:
                    print("[check] Unable to compute top zones.")
        else:
            if verbose:
                print("[check] No 'Zone' column (case-insensitive) found.")

        return df
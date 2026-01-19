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



    # ----- lifecycle -----
    def __init__(self, *, verbose: int = 1, out_dir: str | None =None):
        print("sefsf")

  
 
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
        Probe + persistent Kalman/Extended Kalman filtering over zones, with a **pluggable
        preparer** for the state/measurement model. This augments the raw snapshot from
        `probe_zone_air_and_supply(...)` with a per-zone EKF update and **persists** both
        measurements and estimates to SQLite.

        High-level flow
        ---------------
        1) Calls `probe_zone_air_and_supply(s, **probe_kwargs)` to get a fast snapshot:
        outdoor (T, w, CO₂) and, per zone, air + supply aggregates.
        2) Applies a **measurement policy** (forward-fill and fallbacks) to produce a clean
        3-vector `y = [T, w, CO2]` for each zone, together with simple regressors
        describing supply/outdoor deltas.
        3) Delegates to a **preparer** (configurable) that maps `(y, mu_prev, P_prev, …)`
        into EKF inputs: `{x_prev, P_prev, f_x, F, H, Q, R, y}`.
        4) Runs a single EKF update via `self._ekf_update(...)`, stores the posterior per
        zone, and **writes** a row to SQLite (batched) with y, ŷ, and μ.

        Measurement policy & fallbacks
        ------------------------------
        • Outdoor & zone readings are **forward-filled**; when first seen, default to 0.  
        • Zone humidity ratio **w** fallback chain:
            1) use probe payload's `w` if finite;
            2) else try *Zone Mean Air Humidity Ratio*;
            3) else compute from `(T, RH, site P)` using a Tetens-based relation.
        • Supply aggregates (mass-flow weighted) and outdoor/supply CO₂ are forward-filled.
        • A single model time step is interpreted as `dt_h = 1/num_time_steps_in_hour`.

        Preparer contract (pluggable model)
        -----------------------------------
        Choose the preparer with `opts["kf_prepare_fn"]` (callable). If omitted, defaults to
        `self._kf_prepare_inputs_zone_energy_model`.

        The preparer is called as either:
            fn(self, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R) -> dict
        or (if bound method):
            fn(*, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R) -> dict

        It must return a dict for `_ekf_update(...)` with keys:
            {
            "x_prev": (n,),               # prior mean (may equal mu_prev)
            "P_prev": (n,n),              # prior covariance (may equal P_prev)
            "f_x":   callable or array,   # nonlinear transition f(x) or predicted x_k|k-1
            "F":     (n,n),               # d f / d x  (Jacobian)
            "H":     (m,n),               # observation matrix
            "Q":     (n,n),               # process noise
            "R":     (m,m),               # measurement noise
            "y":     (m,)                 # measurement vector (usually [T,w,CO2])
            }

        Notes on the built-in measurement features
        ------------------------------------------
        • The function constructs a simple regressor matrix `phi` based on supply/outdoor deltas
        and bundles a rich `meas` dict for the preparer:
            `meas = {"phi", "y", "names": ["T","w","CO2"], "ts", "dt", "msa", "To","wo","co","Tsa","wsa","csa"}`
        (`msa` = supply mass flow, kg/s)
        • State dimensionality **n** is determined by your preparer (in practice inferred from
        `mu_prev` / `mu0` length; defaults below imply n=5).

        Persistence (SQLite)
        --------------------
        • Database file: `<out_dir>/<kf_db_filename>` (default: `"eplusout.sql"`).  
        This function adds a table alongside EnergyPlus tables (safe to co-exist).
        • Table name: `opts["kf_sql_table"]` (default: `"KalmanEstimates"`).
        • Base schema (created if absent):

            Timestamp TEXT, Zone TEXT,
            y_T REAL, y_w REAL, y_c REAL,
            yhat_T REAL, yhat_w REAL, yhat_c REAL

        On first insert, **state columns** are added dynamically as `mu_0, mu_1, …`.
        (Optionally set `self._kf_state_col_names = ["Qint","Cth","..."]` beforehand to
        override column names.)
        • Batched writes with WAL by default; commits/checkpoints are tunable (see options).

        Parameters
        ----------
        s : EnergyPlusState
            Active runtime state (provided by the callback hook).
        **opts :
            The following keys are recognized (others are forwarded to `probe_zone_air_and_supply`):

            Kalman model / noise / priors
            • `kf_sigma_P_diag`: sequence[float]  
            Process noise diagonal (default: `[1e-6, 1e-3, 1e-6, 1e-6, 1e-4]`).
            • `kf_sigma_R_diag`: sequence[float]  
            Measurement noise diagonal for `[T, w, CO2]` (default: `[0.2**2, (2e-4)**2, 30.0**2]`).
            • `kf_init_mu`: sequence[float]  
            Initial mean μ₀ (default: `[0.0, 20.0, 0.0, 0.008, 400.0]`).
            • `kf_init_cov_diag`: sequence[float]  
            Initial covariance diag (default: `[1.0, 25.0, 1.0, 1e-3, 1e3]`).
            • `kf_prepare_fn`: callable  
            Preparer function/method as described above (default: `_kf_prepare_inputs_zone_energy_model`).

            Zone selection & logging
            • `kf_zones`: list[str] | None  
            If provided, restricts filtering/updates to these zones (others ignored).
            • `kf_exclude_patterns`: tuple[str,...]  
            Exclude zones containing any of these substrings (default: `("PLENUM",)`).
            • `kf_log`: bool  
            Log a concise per-update line (default: True).

            Persistence & performance
            • `kf_db_filename`: str  
            SQLite file name (default: `"eplusout.sql"`).
            • `kf_sql_table`: str  
            Table name (default: `"KalmanEstimates"`).
            • `kf_batch_size`: int  
            Insert batch size for `executemany` (default: 50).
            • `kf_commit_every_batches`: int  
            Commit after this many batches (default: 10).
            • `kf_checkpoint_every_commits`: int  
            For WAL, checkpoint after this many commits (default: 5).
            • `kf_journal_mode`: {"WAL","DELETE","TRUNCATE","MEMORY","OFF"} (default: "WAL")
            • `kf_synchronous`: {"OFF","NORMAL","FULL","EXTRA"} (default: "NORMAL")

            Forwarded to `probe_zone_air_and_supply(...)`
            • Any keys not listed above (e.g., `log_every_minutes`, `precision`, …) go directly to
            the underlying probe.

        Returns
        -------
        dict | None
            The **probe payload** (same structure as `probe_zone_air_and_supply`), or `None` during
            warmup / before handles are ready / if the probe yields nothing. EKF posteriors are stored
            internally on:
                • `self.__dict__["_kf_mu"][zone]    → numpy.ndarray (n,)`
                • `self.__dict__["_kf_Sigma"][zone] → numpy.ndarray (n,n)`
            and rows are persisted to SQLite as described above.

        Side effects
        ------------
        • Maintains per-run caches for forward-fill:
            `_kf_last_out`, `_kf_last_air[zone]`, `_kf_last_sup[zone]`.  
        • Opens/creates `<out_dir>/<kf_db_filename>`, creates/extends `kf_sql_table`, and writes
        batched inserts (with WAL/synchronous pragmas if supported).
        • Logs a compact line per zone when `kf_log=True`, e.g.:
            `[kf] 2002-01-01 01:00:00 LIVING | update | yhat_T=21.8 yhat_w=0.00791 yhat_c=420.5 | mu[:5]=[...]`

        Error handling & edge cases
        ---------------------------
        • Returns early during EnergyPlus warmup.  
        • If SQLite fails to open/insert, persistence is disabled for the remainder of the run
        (probe payload is still returned).  
        • If no zones pass the selection/filtering, nothing is updated/persisted.  
        • The default `kf_db_filename="eplusout.sql"` appends a custom table to the EnergyPlus
        database; to keep KF data separate, set a different filename (e.g., `"kalman.sqlite"`).

        Examples
        --------
        1) **Register with defaults** (hourly logging from the probe suppressed, KF logging on):
            >>> util.register_begin_iteration([
            ...   {"method_name": "probe_zone_air_and_supply_with_kf",
            ...    "kwargs": {"log_every_minutes": None, "kf_log": True}}
            ... ])
            >>> util.run_annual()

        2) **Restrict to a few zones** and write KF outputs to a separate DB:
            >>> util.register_begin_iteration([
            ...   {"method_name": "probe_zone_air_and_supply_with_kf",
            ...    "kwargs": {
            ...       "kf_zones": ["LIVING", "KITCHEN"],
            ...       "kf_db_filename": "kalman.sqlite",
            ...       "kf_sql_table": "ZoneEKF",
            ...       "kf_sigma_R_diag": [0.25**2, (3e-4)**2, 20.0**2]
            ...    }}
            ... ])
            >>> util.run_design_day()

        3) **Custom preparer** (e.g., random-walk state with direct mapping):
            >>> def my_prepare(self, *, zone, meas, mu_prev, P_prev, Sigma_P, Sigma_R):
            ...     import numpy as np
            ...     # n = len(mu_prev); m = 3 for [T,w,CO2]
            ...     n = len(mu_prev)
            ...     F = np.eye(n)                      # x_k = x_{k-1} + noise
            ...     H = np.zeros((3, n)); H[:,:3] = np.eye(3)  # observe first 3 states directly
            ...     Q = Sigma_P
            ...     R = Sigma_R
            ...     def f_x(x): return x               # identity transition
            ...     return dict(
            ...         x_prev=mu_prev, P_prev=P_prev,
            ...         f_x=f_x, F=F, H=H, Q=Q, R=R, y=meas["y"]
            ...     )
            ...
            >>> util.register_begin_iteration([
            ...   {"method_name": "probe_zone_air_and_supply_with_kf",
            ...    "kwargs": {"kf_prepare_fn": my_prepare, "kf_log": True}}
            ... ])
            >>> util.run_annual()

        4) **Query the persisted estimates** (after a run):
            >>> import sqlite3, pandas as pd, os
            >>> db = os.path.join(util.out_dir, "kalman.sqlite")  # or "eplusout.sql" if you used the default
            >>> conn = sqlite3.connect(db)
            >>> df = pd.read_sql_query("SELECT * FROM ZoneEKF WHERE Zone='LIVING' ORDER BY Timestamp", conn)
            >>> conn.close()
            >>> df.head()

        Implementation notes
        --------------------
        • Default process/measurement covariances and priors are **reasonable starting points**
        but should be tuned for your building and sensor quality.  
        • The EKF step relies on `self._ekf_update(...)`. Your preparer determines the model; the
        helper defaults here imply a 5-state example (because default μ₀/Σ₀ diagonals are length 5),
        but nothing in the code constrains you to that size.
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

    # --- Control ----

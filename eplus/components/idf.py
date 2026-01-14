import os
import shutil
from typing import Optional

class IDFMixin:
    def __init__(self):
        self.idf: Optional[str] = None
        self.epw: Optional[str] = None
        self.out_dir: Optional[str] = None
        self._patched_idf_path: Optional[str] = None
        self._orig_idf_path: Optional[str] = None

    def set_model(self, idf: str, epw: str, out_dir: Optional[str] = None, *, reset: bool = True, **kwargs) -> None:
        """
        Configure the active EnergyPlus model paths and (optionally) inject a minimal
        CO₂ setup, ready for subsequent runs.

        This sets `self.idf`, `self.epw`, and `self.out_dir` (creating the output
        directory if needed). If `reset=True`, the EnergyPlus state is reset so that
        subsequent runs start clean.

        If `add_co2=True`, this calls `prepare_run_with_co2(...)` to:
        - enable zone CO₂ accounting via `ZoneAirContaminantBalance`,
        - create/bind an **outdoor CO₂ schedule** seeded to `outdoor_co2_ppm`,
        - patch each `People` object with a **CO₂ generation rate coefficient**
            (`per_person_m3ps_per_W`, in m³·s⁻¹ per W per person),
        - write a patched IDF in `out_dir` and switch `self.idf` to that file.
        (That helper also resets state by default, so the model will be ready to run
        with the CO₂ features active.)

        Parameters
        ----------
        idf : str
            Path to the IDF model to load.
        epw : str
            Path to the EPW weather file to use.
        out_dir : Optional[str], default None
            Directory for EnergyPlus outputs; created if missing. Defaults to
            ``"eplus_out"`` when not provided.
        reset : bool, default True
            If True, reset the EnergyPlus API state immediately after setting paths.
        add_co2 : bool, default True
            If True, inject the minimal CO₂ workflow via `prepare_run_with_co2(...)`
            and switch `self.idf` to the patched file.
        outdoor_co2_ppm : float, default 420.0
            Initial value for the outdoor CO₂ schedule (ppm) when `add_co2=True`.
        per_person_m3ps_per_W : float, default 3.82e-8
            People CO₂ generation coefficient (m³/s per W per person). EnergyPlus’s
            default is 3.82e-8; values are clamped to the model’s allowed range
            inside the helper.

        Notes
        -----
        - This method **does not run** a simulation; it only configures paths/state.
        - When `add_co2=True`, `self._orig_idf_path` is remembered and `self.idf`
        points to the newly written CO₂-patched IDF in `out_dir`.

        Returns
        -------
        None

        Examples
        --------
        Basic setup with CO₂ enabled (default):
        >>> util.set_model("models/small_office.idf", "weather/USA_CA_San-Francisco.epw",
        ...                out_dir="runs/run1")

        Custom outdoor CO₂ and generation rate:
        >>> util.set_model("bldg.idf", "site.epw", out_dir="out",
        ...                add_co2=True, outdoor_co2_ppm=450.0,
        ...                per_person_m3ps_per_W=3.5e-8)

        Skip CO₂ patching entirely:
        >>> util.set_model("bldg.idf", "site.epw", out_dir="out", add_co2=False)
        """
        self.idf = str(idf)
        self.epw = str(epw)
        self.out_dir = str(out_dir or "eplus_out")
        
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Call reset_state from StateMixin via self
        if reset and hasattr(self, 'reset_state'):
            self.reset_state()
        
        # Logic for CO2 (Check if the CO2 Mixin is present)
        if kwargs.get("add_co2", False) and hasattr(self, "prepare_run_with_co2"):
            self.prepare_run_with_co2(
                outdoor_co2_ppm=kwargs.get("outdoor_co2_ppm", 420.0),
                per_person_m3ps_per_W=kwargs.get("per_person_m3ps_per_W", 3.82e-8)
            )

    def delete_out_dir(self):
        """
        Delete the output directory (`self.out_dir`) and all of its contents, if it exists.

        The directory is removed recursively via `shutil.rmtree(..., ignore_errors=True)`.
        Missing directories or removal errors are silently ignored. This only affects the
        on-disk folder; the `self.out_dir` attribute is not modified.
        """        
        import shutil, os
        if self.out_dir and os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir, ignore_errors=True)
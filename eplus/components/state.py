# eplus/mixins/state.py
from pyenergyplus.api import EnergyPlusAPI
from typing import Optional
import os
import shutil
import urllib.request
from urllib.parse import urlparse
import glob

class StateMixin(EnergyPlusAPI):
    """
    Inherits from EnergyPlusAPI so the main class allows direct access 
    to API sub-modules like self.exchange, self.runtime, etc.
    """
    def __init__(self):
        self._log(2, "Initialized StateMixin")
        # Initialize the parent EnergyPlusAPI
        # This sets up self.state_manager, self.runtime, self.exchange, etc.
        super().__init__()
        
        self.idf: Optional[str] = None
        self.epw: Optional[str] = None
        self.out_dir: Optional[str] = getattr(self, 'out_dir', "eplus_out")

        # Create the new state using the inherited state_manager
        self.state = self.state_manager.new_state()
        self._log(2, "Initialized new EnergyPlus state.")

    def reset_state(self) -> None:
        """
        Resets the EnergyPlus C++ state. 
        Accesses logging methods from LoggingMixin via self.
        """
        try:
            if getattr(self, "state", None):
                # Use self.state_manager directly (inherited from EnergyPlusAPI)
                self.state_manager.reset_state(self.state)
        except Exception:
            pass
        
        self.state = self.state_manager.new_state()
        
        # Re-attach logger if it exists (assuming LoggingMixin is present)
        if getattr(self, "_runtime_log_enabled", False) and getattr(self, "_runtime_log_func", None):
            # Use self.runtime directly (inherited from EnergyPlusAPI)
            self.runtime.callback_message(self.state, self._runtime_log_func)
        self._log(1, "EnergyPlus state has been reset.")

    def clear_eplus_outputs(self, patterns: tuple[str, ...] = ("eplusout.*",)) -> None:
        """
        Remove common EnergyPlus outputs in out_dir, especially a stale/locked eplusout.sql.
        Safe to call before runs.
        """
        assert self.out_dir, "set_model(...) first."
        for pat in patterns:
            for p in glob.glob(os.path.join(self.out_dir, pat)):
                try: 
                    os.remove(p)
                    self._log(1, f"Deleted output file: {p}")
                except IsADirectoryError: pass
                except FileNotFoundError: pass
                except PermissionError: pass  # leave it if OS blocks; at least we tried

    def delete_out_dir(self):
        """
        Delete the output directory (`self.out_dir`) and all of its contents, if it exists.

        The directory is removed recursively via `shutil.rmtree(..., ignore_errors=True)`.
        Missing directories or removal errors are silently ignored. This only affects the
        on-disk folder; the `self.out_dir` attribute is not modified.
        """        
        if self.out_dir and os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self._log(1, f"Deleted output directory: {self.out_dir}")
        else:
            self._log(2, f"Output directory does not exist, nothing to delete: {self.out_dir}")

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
        if out_dir is not None:
            self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Call reset_state from StateMixin via self
        if reset and hasattr(self, 'reset_state'):
            self.reset_state()
        self._log(1, f"Model set: IDF='{self.idf}', EPW='{self.epw}', OUT_DIR='{self.out_dir}'")

        # TODO: handle CO2 mixin logic here
        # Logic for CO2 (Check if the CO2 Mixin is present)
        # if kwargs.get("add_co2", False) and hasattr(self, "prepare_run_with_co2"):
        #     self.prepare_run_with_co2(
        #         outdoor_co2_ppm=kwargs.get("outdoor_co2_ppm", 420.0),
        #         per_person_m3ps_per_W=kwargs.get("per_person_m3ps_per_W", 3.82e-8)
        #     )

    def set_model_from_url(self, idf_url: str, epw_url: str, out_dir: Optional[str] = None, **kwargs) -> None:
        """
        Downloads IDF and EPW files from URLs and configures the model.
        
        This downloads the files into `out_dir` (or "eplus_out") and then calls 
        `set_model` with the local paths. It passes all extra arguments (like `add_co2`) 
        through to `set_model`.

        Parameters
        ----------
        idf_url : str
            Direct URL to the .idf file (e.g. GitHub raw link).
        epw_url : str
            Direct URL to the .epw file.
        out_dir : str, optional
            Directory to save files and run outputs. Defaults to "eplus_out".
        **kwargs : 
            Passed directly to `set_model` (e.g. `add_co2=True`).
        """
        if out_dir is not None:
            self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        target_dir = self.out_dir

        # Helper to get filename from URL
        def download_file(url, directory):
            # Parse filename from URL (e.g. ".../Model.idf" -> "Model.idf")
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename: 
                filename = "downloaded_model.idf" if "idf" in url else "weather.epw"
            
            local_path = os.path.join(directory, filename)
            
            if hasattr(self, '_log'):
                self._log(1, f"Downloading {filename} from {url}...")
            
            # Download using standard library (works on Win/Linux/Mac)
            urllib.request.urlretrieve(url, local_path)
            return local_path

        try:
            local_idf = download_file(idf_url, target_dir)
            local_epw = download_file(epw_url, target_dir)
            
            # Delegate to the main set_model method
            self.set_model(local_idf, local_epw, out_dir=target_dir, **kwargs)

        except Exception as e:
            if hasattr(self, '_log'):
                self._log(1, f"Error downloading model files: {e}")
            raise e
        
        # TODO: handle CO2 mixin logic here
        # Logic for CO2 (Check if the CO2 Mixin is present)
        # if kwargs.get("add_co2", False) and hasattr(self, "prepare_run_with_co2"):
        #     self.prepare_run_with_co2(
        #         outdoor_co2_ppm=kwargs.get("outdoor_co2_ppm", 420.0),
        #         per_person_m3ps_per_W=kwargs.get("per_person_m3ps_per_W", 3.82e-8)
        #     )
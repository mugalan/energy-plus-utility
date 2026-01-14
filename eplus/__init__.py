# eplus/__init__.py
import warnings

# 1. Import the bootstrap function (Safe: has no external dependencies)
from .colab_bootstrap import prepare_colab_eplus

__version__ = "0.2.0+1"
__all__ = ["prepare_colab_eplus", "__version__"]

# 2. Try to import the Core Logic
# This block will fail on a fresh Colab instance because 'pyenergyplus' 
# isn't installed yet. We catch the error so 'prepare_colab_eplus' 
# remains accessible.
try:
    from .core import EPlusUtil

    class EPlusUtil(EPlusUtil):
        """
        Utility wrapper around pyenergyplus EnergyPlusAPI.
        
        • State is created at init and is resettable.
        • set_model(idf, epw, out_dir)
        • run_design_day() / run_annual() 
        """
        pass

    # If import succeeded, add EPlusUtil to exported names
    __all__.append("EPlusUtil")

except ImportError:
    # This block runs if pyenergyplus is missing.
    # We silently fail here so the user can still run:
    #   from eplus import prepare_colab_eplus
    #   prepare_colab_eplus()
    pass
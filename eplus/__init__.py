# eplus/__init__.py
from .colab_bootstrap import prepare_colab_eplus  # safe to import immediately
__all__ = ["prepare_colab_eplus", "EPlusUtil", "__version__"]
__version__ = "0.1.0"

# Lazy export to avoid importing pyenergyplus until AFTER bootstrap is run
def __getattr__(name):
    if name == "EPlusUtil":
        from .eplus_util import EPlusUtil  # imports pyenergyplus here, after bootstrap
        return EPlusUtil
    raise AttributeError(f"module 'eplus' has no attribute {name!r}")

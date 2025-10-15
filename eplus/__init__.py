from .colab_bootstrap import prepare_colab_eplus
from .sql_explorer import EPlusSqlExplorer

# Lazy-load EPlusUtil to avoid requiring pyenergyplus before bootstrap
__version__ = "0.1.0"
__all__ = ["prepare_colab_eplus", "EPlusSqlExplorer", "EPlusUtil", "__version__"]

def __getattr__(name):
    if name == "EPlusUtil":
        from .eplus_util import EPlusUtil
        return EPlusUtil
    raise AttributeError(f"module 'eplus' has no attribute {name!r}")

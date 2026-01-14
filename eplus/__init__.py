# eplus/__init__.py
from .colab_bootstrap import prepare_colab_eplus
from .core import BaseEPlusUtil


__version__ = "0.2.1"

class EPlusUtil(
    BaseEPlusUtil,
):
    """
    Utility wrapper around pyenergyplus EnergyPlusAPI.

    • State is created at init and is resettable.
    • set_model(idf, epw, out_dir)
    • run_design_day() / run_annual() with centralized callback registration
    • list_variables_safely() / list_actuators_safely() via tiny design-day runs
    • set_simulation_params() patches timestep/runperiod (writes ...__patched.idf)

    SQL-focused plotting/IO:
      - plot_sql_series(...)
      - plot_sql_meters(...)
      - plot_sql_zone_variable(...)
      - plot_sql_net_purchased_electricity(...)
      - export_weather_sql_to_csv(...)

    Notes:
      - reporting_freq=None in plotters means "no frequency filter" (passes through any freq present).
    """
    pass

__all__ = ["prepare_colab_eplus", "EPlusUtil", "__version__"]

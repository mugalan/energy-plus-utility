# eplus/core.py
# Import your mixins
from .components.state import StateMixin
from .components.idf import IDFMixin
from .components.logging import LoggingMixin
from .components.simulation import SimulationMixin
from .components.utils import UtilsMixin
from .components.handlers import HandlersMixin
from .components.sql import SQLMixin
from .components.control import ControlMixin
from .components.occupancy import OccupancyMixin
from .components.zone_observer import ZoneObserverMixin

class EPlusUtil(StateMixin, IDFMixin, LoggingMixin, SimulationMixin, 
                UtilsMixin, HandlersMixin, SQLMixin, ControlMixin,
                OccupancyMixin, ZoneObserverMixin):
    """
    Main class that combines State, IDF, and Logging functionalities.
    Inheritance order matters: Methods in StateMixin are checked before IOMixin, etc.
    """
    
    def __init__(self, *, verbose: int = 1, out_dir: str | None = None):
        # Initialize attributes specifically for this class
        self.verbose = int(verbose)
        self.out_dir = str(out_dir)
        
        # Initialize Mixins manually to ensure all attributes are set up.
        StateMixin.__init__(self)
        IDFMixin.__init__(self)
        LoggingMixin.__init__(self)
        SimulationMixin.__init__(self)
        UtilsMixin.__init__(self)
        HandlersMixin.__init__(self)
        SQLMixin.__init__(self)
        ControlMixin.__init__(self)
        OccupancyMixin.__init__(self)
        ZoneObserverMixin.__init__(self)


        # Override IO default if provided
        if out_dir:
            self.out_dir = out_dir

    # You can add methods here that specifically need to coordinate 
    # between all three mixins, or leave this class empty.
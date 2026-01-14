# eplus/mixins/state.py
from pyenergyplus.api import EnergyPlusAPI

class StateMixin(EnergyPlusAPI):
    """
    Inherits from EnergyPlusAPI so the main class allows direct access 
    to API sub-modules like self.exchange, self.runtime, etc.
    """
    def __init__(self):
        # Initialize the parent EnergyPlusAPI
        # This sets up self.state_manager, self.runtime, self.exchange, etc.
        super().__init__()
        
        # Create the new state using the inherited state_manager
        self.state = self.state_manager.new_state()

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
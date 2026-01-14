from typing import List, Tuple, Callable

class LoggingMixin:
    def __init__(self):
        self._log(2, "Initialized LoggingMixin")
        self.verbose: int = getattr(self, 'verbose', 1)

    def _log(self, level: int, msg: str):
        if self.verbose >= level:
            print(msg)
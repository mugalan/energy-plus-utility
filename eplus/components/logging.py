from typing import List, Tuple, Callable

class LoggingMixin:
    def __init__(self):
        self.verbose: int = getattr(self, 'verbose', 1)
        self._runtime_log_enabled: bool = False
        self._runtime_log_func: Callable = None
        self._extra_callbacks: List[Tuple[Callable, Callable]] = []

    def _log(self, level: int, msg: str):
        if self.verbose >= level:
            print(msg)
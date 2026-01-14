import os
import shutil
from typing import Optional

class IDFMixin:
    def __init__(self):
        self._patched_idf_path: Optional[str] = None
        self._orig_idf_path: Optional[str] = None




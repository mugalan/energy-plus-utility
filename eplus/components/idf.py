import os
import shutil
from typing import Optional
import os, io, csv as _csv, ast, shutil, pathlib, subprocess, re, tempfile, contextlib

class IDFMixin:
    def __init__(self):
        self._log(2, "Initialized IDFMixin")
        self._patched_idf_path: Optional[str] = None
        self._orig_idf_path: Optional[str] = None
    
    # TODO:Add to ColabDOCs
    def clear_patched_idf(self):
        """Revert to the original IDF if we switched to a patched one."""
        if getattr(self, "_orig_idf_path", None):
            self.idf = self._orig_idf_path
        self._patched_idf_path = None
        self._orig_idf_path = None  # also clear to return to a clean slate

    def _remove_object_blocks(self, idf_text: str, obj_name: str) -> str:
        """Remove ALL blocks of a given object (case-insensitive; simple regex parser)."""
        pattern = rf'(?is)^\s*{re.escape(obj_name)}\s*,.*?;[ \t]*\n'
        return re.sub(pattern, '', idf_text, flags=re.MULTILINE)

    def _append_block(self, idf_text: str, block: str) -> str:
        return idf_text.rstrip() + "\n\n" + block.strip() + "\n"


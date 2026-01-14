import os, tempfile, pathlib

class UtilsMixin:
    def __init__(self):
        self._log(2, "Initialized UtilsMixin")
        
    def _assert_out_dir_writable(self):
        assert self.out_dir, "set_model(...) first."
        os.makedirs(self.out_dir, exist_ok=True)
        # quick write test
        tmp = pathlib.Path(self.out_dir) / ".write_test.tmp"
        with open(tmp, "wb") as f:
            f.write(b"ok")
        tmp.unlink(missing_ok=True)

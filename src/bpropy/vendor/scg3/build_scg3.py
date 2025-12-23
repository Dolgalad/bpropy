# src/bpropy/vendor/scg3/build_scg3.py
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        super().run()

        # This file lives in: src/bpropy/vendor/scg3/build_scg3.py
        here = Path(__file__).resolve().parent
        c_file = here / "scg3.c"
        if not c_file.exists():
            raise FileNotFoundError(f"scg3.c not found at {c_file}")

        exe_name = "scg3.exe" if os.name == "nt" else "scg3"

        # 1) Always build into build_lib so wheels contain it
        wheel_out_dir = Path(self.build_lib) / "bpropy" / "vendor" / "scg3"
        wheel_out_dir.mkdir(parents=True, exist_ok=True)
        wheel_exe = wheel_out_dir / exe_name

        # 2) If editable, also build into the source tree so imports can see it
        editable = bool(getattr(self, "editable_mode", False))
        src_exe = here / exe_name  # next to scg3.c in src tree

        cc = os.environ.get("CC", "gcc")
        cmd = [cc, "-O2", "-Wall", str(c_file), "-o", str(wheel_exe), "-lm"]
        subprocess.check_call(cmd)

        # ensure executable bit on unix
        if os.name != "nt":
            wheel_exe.chmod(wheel_exe.stat().st_mode | 0o111)

        if editable:
            # copy/overwrite into src tree
            src_exe.write_bytes(wheel_exe.read_bytes())
            if os.name != "nt":
                src_exe.chmod(src_exe.stat().st_mode | 0o111)


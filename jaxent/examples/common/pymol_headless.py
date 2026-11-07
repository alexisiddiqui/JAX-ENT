"""Headless execution of the ``jaxent/src/analysis/pymol`` scripts.

PyMOL executes a ``.py`` named on its command line in the ``pymol`` namespace (``__name__ == "pymol"``), so a
script's ``if __name__ == "__main__":`` block is skipped; ``run script, main`` fixes the namespace but replaces
``sys.argv`` with PyMOL's own.  This module is both the shim that PyMOL runs and the Python-side launcher:

    pymol -cq pymol_headless.py -- <script.py> [script args ...]      (what run_pymol_script executes)

    from jaxent.examples.common.pymol_headless import run_pymol_script
    run_pymol_script(REPO/"jaxent/src/analysis/pymol/Project_logPFs.py",
                     ["--config", cfg, "--working_dir", wd, "--save_png", "--output_prefix", stem])

The PyMOL binary is ``$JAXENT_PYMOL`` if set, else the ``PYMOL_ANA`` conda environment's ``pymol``, else
``pymol`` on PATH.  ``width``/``height`` set the off-screen viewport (``cmd.viewport``; PyMOL ignores -W/-H in
``-c`` mode) and hence the ray-traced PNG size.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

_CANDIDATES = (
    os.environ.get("JAXENT_PYMOL"),
    os.path.expanduser("~/anaconda3/envs/PYMOL_ANA/bin/pymol"),
    os.path.expanduser("~/miniconda3/envs/PYMOL_ANA/bin/pymol"),
    shutil.which("pymol"),
)


def find_pymol() -> str | None:
    for c in _CANDIDATES:
        if c and Path(c).exists():
            return str(c)
    return None


def run_pymol_script(script, args, pymol=None, width=1600, height=1200, cwd=None, log=None):
    """Run ``script`` (a jaxent pymol script) as ``__main__`` inside headless PyMOL.

    Returns the CompletedProcess; raises FileNotFoundError when no PyMOL binary is found and
    CalledProcessError on a non-zero exit.  stdout/stderr are captured (and appended to ``log`` if given).
    """
    pymol = pymol or find_pymol()
    if pymol is None:
        raise FileNotFoundError("no PyMOL binary (set JAXENT_PYMOL or create the PYMOL_ANA conda env)")
    cmd = [pymol, "-cq", str(Path(__file__).resolve()), "--", str(Path(script).resolve()), *map(str, args)]
    env = {**os.environ, "JAXENT_PYMOL_VIEWPORT": f"{int(width)}x{int(height)}"}
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if log is not None:
        with open(log, "a") as fh:
            fh.write("$ "+" ".join(cmd)+"\n"+proc.stdout+proc.stderr+"\n")
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


if __name__ == "pymol":  # executed by PyMOL: pymol -cq pymol_headless.py -- <script> [args]
    _script, *_args = sys.argv[1:]
    sys.argv = [_script] + _args
    if os.environ.get("JAXENT_PYMOL_VIEWPORT"):
        from pymol import cmd as _cmd
        _cmd.viewport(*(int(v) for v in os.environ["JAXENT_PYMOL_VIEWPORT"].split("x")))
    exec(compile(open(_script).read(), _script, "exec"), {"__name__": "__main__", "__file__": _script})

import os
import sys

print("--- DEBUG ---")
print(f"__file__: {__file__ if '__file__' in globals() else 'MISSING'}")
print(f"__file__ (absolute): {os.path.abspath(__file__) if '__file__' in globals() else 'N/A'}")
print(f"CWD: {os.getcwd()}")

try:
    _raw_file = __file__
except NameError:
    _raw_file = "NOT_DEFINED"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(_raw_file))
print(f"Computed SCRIPT_DIR: {_SCRIPT_DIR}")
print(f"Does utils.py exist in SCRIPT_DIR? {os.path.exists(os.path.join(_SCRIPT_DIR, 'utils.py'))}")

if _SCRIPT_DIR not in sys.path:
    print(f"Adding {_SCRIPT_DIR} to sys.path")
    sys.path.insert(0, _SCRIPT_DIR)
else:
    print(f"{_SCRIPT_DIR} already in sys.path")

print(f"Final sys.path[0:3]: {sys.path[:3]}")

try:
    import utils
    print("SUCCESS: imported utils")
except ImportError as e:
    print(f"FAILURE: could not import utils: {e}")

try:
    from pymol import cmd
    print("SUCCESS: imported pymol.cmd")
except ImportError as e:
    print(f"FAILURE: could not import pymol.cmd: {e}")

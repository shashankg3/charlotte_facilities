# =============================================================
# V1_scaling_benchmark_fl.py  (Food Lion study)
#
# Runtime scaling benchmark for the Food Lion case study.
# The road graph and Multi-source Dijkstra are IDENTICAL to the
# Harris Teeter study -- V1 measures Dijkstra timing only, which
# is brand-independent.
#
# This script re-runs the same benchmark as validations/V1_scaling_benchmark.py
# and saves output to Results_final/FL_results/V1/
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations_foodlion/V1/V1_scaling_benchmark_fl.py --charlotte --ram-gb 8
#
# On HPC with 80 GB:
#   python validations_foodlion/V1/V1_scaling_benchmark_fl.py --all --ram-gb 80
# =============================================================

import os, sys

# Point output dir to FL results
os.environ.setdefault("V1_OUTPUT_DIR", "Results_final/FL_results/V1")
os.makedirs(os.environ["V1_OUTPUT_DIR"], exist_ok=True)

# Re-use the original V1 script wholesale -- it is brand-independent
# (measures road graph timing, not store scoring)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validations"))

# Run the original benchmark -- just redirect its default output dir
import importlib.util, pathlib

orig = pathlib.Path(__file__).parent.parent.parent / "validations" / "V1_scaling_benchmark.py"
spec = importlib.util.spec_from_file_location("V1_scaling_benchmark", str(orig))
mod  = importlib.util.module_from_spec(spec)

# Patch output directory before executing
import builtins
_orig_open = builtins.open

def _patched_open(file, mode="r", *a, **kw):
    if isinstance(file, str) and "Results_final/V1_results" in file:
        file = file.replace("Results_final/V1_results",
                            os.environ["V1_OUTPUT_DIR"])
    return _orig_open(file, mode, *a, **kw)

builtins.open = _patched_open
try:
    spec.loader.exec_module(mod)
finally:
    builtins.open = _orig_open

print(f"\n[V1-FL] Results saved to {os.environ['V1_OUTPUT_DIR']}")
print("[V1-FL] Note: Road graph timing is brand-independent.")
print("[V1-FL] Food Lion V1 results are identical to Harris Teeter V1.")

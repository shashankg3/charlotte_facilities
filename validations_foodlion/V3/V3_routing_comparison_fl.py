# =============================================================
# V3_routing_comparison_fl.py  (Food Lion study)
#
# Routing method comparison for the Food Lion case study.
# The road graph, BG set, and candidate set are IDENTICAL to
# the Harris Teeter study -- V3 measures routing method timing
# on the Charlotte network, which is brand-independent.
#
# Results are identical to the HT V3 benchmark.
# This script exists for completeness / reproducibility of the
# Food Lion study, and re-runs the same four-method comparison
# saving output to Results_final/FL_results/V3/.
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations_foodlion/V3/V3_routing_comparison_fl.py
#
# NOTE: OSRM requires Docker. If Docker is not available, only the
# three Python methods will run and OSRM will be skipped.
# =============================================================

import os, sys

# Redirect V3 output to FL results folder
V3_OUT = "Results_final/FL_results/V3"
os.makedirs(V3_OUT, exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# V3 is purely a routing benchmark -- brand-independent.
# Re-use original V3 script logic wholesale.
import importlib.util, pathlib, builtins

orig = pathlib.Path(__file__).parent.parent.parent / "validations" / "V3_OSRM_benchmark.py"

if not orig.exists():
    print(f"[V3-FL] ERROR: original V3 script not found at {orig}")
    sys.exit(1)

# Patch open() so any output files written to V3_results go to FL V3 folder
_orig_open = builtins.open

def _patched_open(file, mode="r", *a, **kw):
    if isinstance(file, str) and "Results_final/V3_results" in file:
        file = file.replace("Results_final/V3_results", V3_OUT)
    return _orig_open(file, mode, *a, **kw)

builtins.open = _patched_open
try:
    spec = importlib.util.spec_from_file_location("V3_OSRM_benchmark", str(orig))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
finally:
    builtins.open = _orig_open

print(f"\n[V3-FL] Results saved to {V3_OUT}")
print("[V3-FL] Note: Routing benchmark is brand-independent.")
print("[V3-FL] Food Lion V3 results are identical to Harris Teeter V3.")

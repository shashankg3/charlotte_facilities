# =============================================================
# V5_hyperparam_search_2_fl.py  (Food Lion study)
#
# IMPROVED hyperparameter search using coordinate descent.
#
# WHY THIS REPLACES hyperparam_search_1:
#   Search 1 found radius/beta/K under balanced weights (0.4/0.3/0.3)
#   then found optimal weights (0.2/0.1/0.7) separately.
#   These two stages contaminate each other — the "best" radius/beta/K
#   under balanced weights may not be best under optimal weights.
#
# THIS APPROACH — Coordinate Descent (2 rounds):
#   Round 1 Stage A — search radius × beta × K  (weights = 0.4/0.3/0.3)
#   Round 1 Stage B — search W1 × W2 × W3       (fixed best r/b/K)
#   Round 2 Stage A — re-search radius × beta × K (weights = Stage B result)
#   Round 2 Stage B — re-search W1 × W2 × W3     (fixed best r/b/K)
#   Converged when Stage A gives same winner both rounds.
#
# GRIDS:
#   radius  : 3, 4, 5, 6, 7  miles
#   beta    : 1.0, 1.5, 2.0, 2.5, 3.0
#   K       : 2, 3, 4, 5
#   W1/W2   : fine grid, W3 = 1 - W1 - W2
#
# OUTPUT:
#   Results_final/FL_results/V5/hyperparam_search_2_results.json
#   Results_final/FL_results/V5/hyperparam_search_2_report.txt
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations_foodlion/V5/V5_hyperparam_search_2_fl.py
# =============================================================

import os, sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_foodlion import load_all, score_all_candidates_like_ht, MILE_M

RESULTS_DIR  = "Results_final/FL_results/V5"
MAX_MATCH_MI = 0.5

# ── Grids ─────────────────────────────────────────────────────
RADIUS_GRID = [3.0, 4.0, 5.0, 6.0, 7.0]
BETA_GRID   = [1.0, 1.5, 2.0, 2.5, 3.0]
K_GRID      = [2, 3, 4, 5]

W1_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
W2_GRID = [0.1, 0.2, 0.3, 0.4]

# ── Core evaluation ───────────────────────────────────────────

def mean_pct_rank(state, radius, beta, K, W1, W2, W3, fl_xy, cand_tree):
    """Return mean percentile rank of matched FL stores. Higher = better."""
    W3 = round(1.0 - W1 - W2, 6)
    if W3 < 0.05:
        return np.nan
    try:
        _, _, _, out_ll = score_all_candidates_like_ht(
            state, radius_miles=radius, beta=beta, K=K,
            W1=W1, W2=W2, W3=W3, return_all=True,
        )
    except Exception:
        return np.nan

    scores     = out_ll["pair_score"].values
    valid_mask = np.isfinite(scores)
    if valid_mask.sum() < 10:
        return np.nan

    valid_scores = scores[valid_mask]
    n_valid      = len(valid_scores)
    rank_order   = np.argsort(np.argsort(valid_scores)).astype(float)
    pct_ranks    = (rank_order / (n_valid - 1)) * 100.0
    pct_full     = np.full(len(scores), np.nan)
    pct_full[valid_mask] = pct_ranks

    fl_dists, fl_idxs = cand_tree.query(fl_xy)
    close   = fl_dists <= (MAX_MATCH_MI * MILE_M)
    matched = fl_idxs[close]
    fl_pcts = pct_full[matched]
    fl_valid = fl_pcts[np.isfinite(fl_pcts)]
    if len(fl_valid) < 5:
        return np.nan
    return float(np.mean(fl_valid))


def search_huff_params(state, W1, W2, W3, fl_xy, cand_tree, label=""):
    """Stage A: search radius × beta × K with fixed weights."""
    combos = list(itertools.product(RADIUS_GRID, BETA_GRID, K_GRID))
    best_score  = -np.inf
    best_params = None
    results     = []
    print(f"\n  [{label}] Searching {len(combos)} combos  "
          f"(W1={W1:.2f} W2={W2:.2f} W3={W3:.2f})")
    t0 = time.perf_counter()
    for i, (r, b, k) in enumerate(combos, 1):
        mpr = mean_pct_rank(state, r, b, k, W1, W2, W3, fl_xy, cand_tree)
        results.append({"radius": r, "beta": b, "K": k,
                        "W1": W1, "W2": W2, "W3": W3,
                        "mean_pct_rank": round(mpr, 3) if np.isfinite(mpr) else None})
        if np.isfinite(mpr) and mpr > best_score:
            best_score  = mpr
            best_params = (r, b, k)
        if i % 20 == 0 or i == len(combos):
            print(f"    {i:3d}/{len(combos)}  "
                  f"best: r={best_params[0]} b={best_params[1]} K={best_params[2]} "
                  f"→ {best_score:.2f}th pctile  [{time.perf_counter()-t0:.0f}s]")
    return best_params, best_score, results


def search_weights(state, radius, beta, K, fl_xy, cand_tree, label=""):
    """Stage B: search W1 × W2 × W3 with fixed Huff params."""
    combos = [(w1, w2, round(1-w1-w2, 4))
              for w1 in W1_GRID for w2 in W2_GRID
              if 0.05 <= round(1-w1-w2, 4) <= 0.75]
    best_score  = -np.inf
    best_w      = None
    results     = []
    print(f"\n  [{label}] Searching {len(combos)} weight combos  "
          f"(r={radius} b={beta} K={K})")
    t0 = time.perf_counter()
    for i, (w1, w2, w3) in enumerate(combos, 1):
        mpr = mean_pct_rank(state, radius, beta, K, w1, w2, w3, fl_xy, cand_tree)
        results.append({"radius": radius, "beta": beta, "K": K,
                        "W1": w1, "W2": w2, "W3": w3,
                        "mean_pct_rank": round(mpr, 3) if np.isfinite(mpr) else None})
        if np.isfinite(mpr) and mpr > best_score:
            best_score = mpr
            best_w     = (w1, w2, w3)
        if i % 10 == 0 or i == len(combos):
            print(f"    {i:3d}/{len(combos)}  "
                  f"best: W1={best_w[0]} W2={best_w[1]} W3={best_w[2]} "
                  f"→ {best_score:.2f}th pctile  [{time.perf_counter()-t0:.0f}s]")
    return best_w, best_score, results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  V5 SEARCH 2: COORDINATE DESCENT — Food Lion")
    print("=" * 65)

    # Load model once
    print("\n[LOAD] Loading Food Lion model state...")
    t0    = time.perf_counter()
    state = load_all()
    fl_m  = state["ht_m"]
    cands = state["cands_m"]
    print(f"       Candidates: {len(cands)}, FL stores: {len(fl_m)}  "
          f"[{time.perf_counter()-t0:.1f}s]")

    cand_xy   = np.c_[cands.geometry.x.values, cands.geometry.y.values]
    cand_tree = cKDTree(cand_xy)
    fl_xy     = np.c_[fl_m.geometry.x.values, fl_m.geometry.y.values]

    all_results = []

    # ── Round 1 ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  ROUND 1")
    print("─" * 65)

    r1a_params, r1a_score, r1a_res = search_huff_params(
        state, 0.4, 0.3, 0.3, fl_xy, cand_tree, "R1-A  Huff params")
    all_results.extend(r1a_res)

    r1b_w, r1b_score, r1b_res = search_weights(
        state, *r1a_params, fl_xy, cand_tree, "R1-B  Weights")
    all_results.extend(r1b_res)

    print(f"\n  Round 1 result:")
    print(f"    Huff: radius={r1a_params[0]} beta={r1a_params[1]} K={r1a_params[2]}")
    print(f"    Weights: W1={r1b_w[0]} W2={r1b_w[1]} W3={r1b_w[2]}")
    print(f"    Score: {r1b_score:.3f}th pctile")

    # ── Round 2 ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  ROUND 2  (re-search with Round 1 weights)")
    print("─" * 65)

    r2a_params, r2a_score, r2a_res = search_huff_params(
        state, *r1b_w, fl_xy, cand_tree, "R2-A  Huff params")
    all_results.extend(r2a_res)

    r2b_w, r2b_score, r2b_res = search_weights(
        state, *r2a_params, fl_xy, cand_tree, "R2-B  Weights")
    all_results.extend(r2b_res)

    print(f"\n  Round 2 result:")
    print(f"    Huff: radius={r2a_params[0]} beta={r2a_params[1]} K={r2a_params[2]}")
    print(f"    Weights: W1={r2b_w[0]} W2={r2b_w[1]} W3={r2b_w[2]}")
    print(f"    Score: {r2b_score:.3f}th pctile")

    # ── Convergence check ─────────────────────────────────────
    converged = (r1a_params == r2a_params)
    print(f"\n  Huff params converged: {'YES ✓' if converged else 'NO — run search_3 with finer grid'}")

    # ── Final result ─────────────────────────────────────────
    best = {
        "radius_miles": r2a_params[0],
        "beta":         r2a_params[1],
        "K":            r2a_params[2],
        "W1":           r2b_w[0],
        "W2":           r2b_w[1],
        "W3":           r2b_w[2],
        "mean_pct_rank": round(r2b_score, 3),
        "converged":    converged,
    }

    print(f"\n{'=' * 65}")
    print(f"  FINAL BEST COMBINATION")
    print(f"{'=' * 65}")
    print(f"  Catchment radius : {best['radius_miles']} miles")
    print(f"  Beta (decay)     : {best['beta']}")
    print(f"  K (competitors)  : {best['K']}")
    print(f"  W1 (Huff demand) : {best['W1']}")
    print(f"  W2 (Dijkstra)    : {best['W2']}")
    print(f"  W3 (Comp penalty): {best['W3']}")
    print(f"  Mean FL pctile   : {best['mean_pct_rank']}th")
    print(f"  Converged        : {'Yes' if converged else 'No — consider search_3'}")
    print(f"{'=' * 65}")

    # ── Round comparison ─────────────────────────────────────
    print(f"\n  Improvement over search_1:")
    print(f"    search_1 best : 63.38th pctile")
    print(f"    search_2 best : {best['mean_pct_rank']}th pctile")
    diff = round(best['mean_pct_rank'] - 63.38, 3)
    print(f"    Δ             : {diff:+.3f}")

    # ── Save ─────────────────────────────────────────────────
    valid_res = [r for r in all_results if r["mean_pct_rank"] is not None]
    top10 = sorted(valid_res, key=lambda x: x["mean_pct_rank"], reverse=True)[:10]

    output = {
        "brand": "Food Lion",
        "city":  "Charlotte, NC",
        "test":  "hyperparam_search_2_coordinate_descent",
        "best_params": best,
        "round1": {
            "huff_params":  {"radius": r1a_params[0], "beta": r1a_params[1], "K": r1a_params[2]},
            "huff_score":   round(r1a_score, 3),
            "weights":      {"W1": r1b_w[0], "W2": r1b_w[1], "W3": r1b_w[2]},
            "weight_score": round(r1b_score, 3),
        },
        "round2": {
            "huff_params":  {"radius": r2a_params[0], "beta": r2a_params[1], "K": r2a_params[2]},
            "huff_score":   round(r2a_score, 3),
            "weights":      {"W1": r2b_w[0], "W2": r2b_w[1], "W3": r2b_w[2]},
            "weight_score": round(r2b_score, 3),
        },
        "converged": converged,
        "top10":     top10,
    }

    json_path = os.path.join(RESULTS_DIR, "hyperparam_search_2_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    report = [
        "V5 HYPERPARAM SEARCH 2 — Coordinate Descent — Food Lion Charlotte",
        "=" * 65,
        "",
        "METHOD: Coordinate descent (2 rounds)",
        "  Round 1A: search radius/beta/K  with W=0.4/0.3/0.3",
        "  Round 1B: search W1/W2/W3       with best r/b/K from 1A",
        "  Round 2A: re-search radius/beta/K with W from 1B",
        "  Round 2B: re-search W1/W2/W3    with best r/b/K from 2A",
        "",
        "ROUND 1",
        f"  Huff params : radius={r1a_params[0]} beta={r1a_params[1]} K={r1a_params[2]}  [{r1a_score:.3f}th pctile]",
        f"  Weights     : W1={r1b_w[0]} W2={r1b_w[1]} W3={r1b_w[2]}  [{r1b_score:.3f}th pctile]",
        "",
        "ROUND 2",
        f"  Huff params : radius={r2a_params[0]} beta={r2a_params[1]} K={r2a_params[2]}  [{r2a_score:.3f}th pctile]",
        f"  Weights     : W1={r2b_w[0]} W2={r2b_w[1]} W3={r2b_w[2]}  [{r2b_score:.3f}th pctile]",
        "",
        f"CONVERGED: {'YES' if converged else 'NO'}",
        "",
        "FINAL BEST COMBINATION",
        f"  radius={best['radius_miles']}mi  beta={best['beta']}  K={best['K']}",
        f"  W1={best['W1']} (Huff demand)   W2={best['W2']} (Dijkstra)   W3={best['W3']} (Comp penalty)",
        f"  Mean FL percentile: {best['mean_pct_rank']}th",
        "",
        "TOP-10",
        f"  {'Rank':<5} {'Radius':>7} {'Beta':>6} {'K':>3} {'W1':>5} {'W2':>5} {'W3':>5} {'MeanPct':>9}",
        "  " + "-" * 55,
    ]
    for i, r in enumerate(top10, 1):
        report.append(
            f"  {i:<5} {r['radius']:>7.1f} {r['beta']:>6.2f} {r['K']:>3} "
            f"{r['W1']:>5.2f} {r['W2']:>5.2f} {r['W3']:>5.2f} "
            f"{r['mean_pct_rank']:>9.3f}"
        )

    txt_path = os.path.join(RESULTS_DIR, "hyperparam_search_2_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report))

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()

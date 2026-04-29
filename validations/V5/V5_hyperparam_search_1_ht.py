# =============================================================
# V5_hyperparam_search_1_ht.py  (Harris Teeter study)
#
# Grid-search over (radius_miles, beta, K, W1, W2, W3) to find
# the combination that best recovers actual Harris Teeter store
# locations — i.e. maximises the mean percentile rank of the
# real HT stores among all scored candidates.
#
# Two-stage search:
#   Stage 1 — coarse grid over radius × beta × K (W fixed at 0.4/0.3/0.3)
#   Stage 2 — fine grid over W1/W2/W3 at the best (radius, beta, K)
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations/V5/V5_hyperparam_search_1_ht.py
#
# OUTPUT:
#   Results_final/V5_results/hyperparam_search_1_results.json
#   Results_final/V5_results/hyperparam_search_1_report.txt
# =============================================================

import os, sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_approach2 import load_all, score_all_candidates_like_ht, MILE_M

RESULTS_DIR  = "Results_final/V5_results"
MAX_MATCH_MI = 0.5          # HT store matched if within 0.5 mi of a candidate

# ── Grid definitions ──────────────────────────────────────────
RADIUS_GRID = [3.0, 4.0, 5.0, 6.0, 7.0]
BETA_GRID   = [1.0, 1.5, 2.0, 2.5, 3.0]
K_GRID      = [2, 3, 4, 5]

# Stage-2 weight grid (must sum to 1; W3 = 1 - W1 - W2)
W1_GRID     = [0.2, 0.3, 0.4, 0.5, 0.6]
W2_GRID     = [0.1, 0.2, 0.3, 0.4]

# ── Helpers ───────────────────────────────────────────────────

def _mean_pct_rank(state, radius, beta, K, W1, W2, W3, ht_xy, cand_tree):
    """
    Score all candidates, match HT stores, return mean percentile rank.
    Higher = HT stores are ranked better by the model.
    Returns NaN if fewer than 5 HT stores can be matched.
    """
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

    # Percentile rank for every valid candidate
    rank_order = np.argsort(np.argsort(valid_scores)).astype(float)
    pct_ranks  = (rank_order / (n_valid - 1)) * 100.0

    pct_full = np.full(len(scores), np.nan)
    pct_full[valid_mask] = pct_ranks

    # Match HT stores → nearest candidate
    ht_dists, ht_idxs = cand_tree.query(ht_xy)
    close     = ht_dists <= (MAX_MATCH_MI * MILE_M)
    matched   = ht_idxs[close]

    ht_pcts = pct_full[matched]
    ht_valid = ht_pcts[np.isfinite(ht_pcts)]
    if len(ht_valid) < 5:
        return np.nan
    return float(np.mean(ht_valid))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  V5: HYPERPARAMETER SEARCH — Harris Teeter")
    print("=" * 65)

    # ── Load model state once ────────────────────────────────
    print("\n[1/3] Loading Harris Teeter model state...")
    t0    = time.perf_counter()
    state = load_all()
    ht_m  = state["ht_m"]
    print(f"      Candidates: {len(state['cands_m'])}, "
          f"HT stores: {len(ht_m)}")
    print(f"      Loaded in {time.perf_counter()-t0:.1f}s")

    # Pre-build KDTree on candidates (reused every evaluation)
    cand_xy   = np.c_[state["cands_m"].geometry.x.values,
                       state["cands_m"].geometry.y.values]
    cand_tree = cKDTree(cand_xy)

    ht_xy = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]

    # ── Stage 1: coarse grid (radius × beta × K) ────────────
    combos_s1 = list(itertools.product(RADIUS_GRID, BETA_GRID, K_GRID))
    n_s1      = len(combos_s1)
    print(f"\n[2/3] Stage 1: coarse grid — {n_s1} combos "
          f"(W1=0.4, W2=0.3, W3=0.3 fixed)")

    best_s1_score  = -np.inf
    best_s1_params = None
    s1_results     = []

    t1 = time.perf_counter()
    for i, (r, b, k) in enumerate(combos_s1, 1):
        mpr = _mean_pct_rank(state, r, b, k, 0.4, 0.3, 0.3, ht_xy, cand_tree)
        s1_results.append({
            "radius": r, "beta": b, "K": k,
            "W1": 0.4, "W2": 0.3, "W3": 0.3,
            "mean_pct_rank": round(mpr, 2) if np.isfinite(mpr) else None,
        })
        if np.isfinite(mpr) and mpr > best_s1_score:
            best_s1_score  = mpr
            best_s1_params = (r, b, k)

        if i % 20 == 0 or i == n_s1:
            elapsed = time.perf_counter() - t1
            print(f"      {i:3d}/{n_s1}  best so far: "
                  f"radius={best_s1_params[0] if best_s1_params else '?'} "
                  f"beta={best_s1_params[1] if best_s1_params else '?'} "
                  f"K={best_s1_params[2] if best_s1_params else '?'} "
                  f"→ {best_s1_score:.1f}th pctile  [{elapsed:.0f}s]")

    best_r, best_b, best_k = best_s1_params
    print(f"\n      Stage 1 winner: radius={best_r} mi, beta={best_b}, K={best_k}")
    print(f"      Mean HT percentile: {best_s1_score:.2f}")

    # ── Stage 2: fine weight search at best (r, b, K) ───────
    w_combos = [(w1, w2, round(1 - w1 - w2, 4))
                for w1 in W1_GRID for w2 in W2_GRID
                if 0.05 <= round(1 - w1 - w2, 4) <= 0.70]
    n_s2 = len(w_combos)
    print(f"\n[3/3] Stage 2: weight search — {n_s2} combos "
          f"(radius={best_r}, beta={best_b}, K={best_k})")

    best_s2_score  = -np.inf
    best_s2_params = None
    s2_results     = []

    t2 = time.perf_counter()
    for i, (w1, w2, w3) in enumerate(w_combos, 1):
        mpr = _mean_pct_rank(state, best_r, best_b, best_k,
                             w1, w2, w3, ht_xy, cand_tree)
        s2_results.append({
            "radius": best_r, "beta": best_b, "K": best_k,
            "W1": w1, "W2": w2, "W3": w3,
            "mean_pct_rank": round(mpr, 2) if np.isfinite(mpr) else None,
        })
        if np.isfinite(mpr) and mpr > best_s2_score:
            best_s2_score  = mpr
            best_s2_params = (w1, w2, w3)

        if i % 10 == 0 or i == n_s2:
            elapsed = time.perf_counter() - t2
            print(f"      {i:3d}/{n_s2}  best so far: "
                  f"W1={best_s2_params[0]} W2={best_s2_params[1]} "
                  f"W3={best_s2_params[2]} → {best_s2_score:.1f}th pctile  "
                  f"[{elapsed:.0f}s]")

    best_w1, best_w2, best_w3 = best_s2_params

    # ── Sort & display top-10 overall ────────────────────────
    all_results = s1_results + s2_results
    valid_res   = [r for r in all_results
                   if r["mean_pct_rank"] is not None]
    top10       = sorted(valid_res,
                         key=lambda x: x["mean_pct_rank"],
                         reverse=True)[:10]

    print(f"\n{'=' * 65}")
    print(f"  BEST COMBINATION FOUND")
    print(f"{'=' * 65}")
    print(f"  Catchment radius : {best_r} miles")
    print(f"  Beta (decay)     : {best_b}")
    print(f"  K (competitors)  : {best_k}")
    print(f"  W1 (gap score)   : {best_w1}")
    print(f"  W2 (income)      : {best_w2}")
    print(f"  W3 (access)      : {best_w3}")
    print(f"  Mean HT pctile   : {best_s2_score:.2f}th")
    print(f"\n  Top-10 configurations:")
    print(f"  {'Rank':<5} {'Radius':>7} {'Beta':>6} {'K':>3} "
          f"{'W1':>5} {'W2':>5} {'W3':>5} {'MeanPct':>9}")
    print(f"  {'-'*55}")
    for i, r in enumerate(top10, 1):
        print(f"  {i:<5} {r['radius']:>7.1f} {r['beta']:>6.2f} "
              f"{r['K']:>3} {r['W1']:>5.2f} {r['W2']:>5.2f} "
              f"{r['W3']:>5.2f} {r['mean_pct_rank']:>9.2f}")
    print(f"{'=' * 65}")

    # ── Save results ─────────────────────────────────────────
    best_params = {
        "radius_miles": best_r,
        "beta":         best_b,
        "K":            best_k,
        "W1":           best_w1,
        "W2":           best_w2,
        "W3":           best_w3,
        "mean_pct_rank": round(best_s2_score, 2),
    }

    output = {
        "brand":          "Harris Teeter",
        "city":           "Charlotte, NC",
        "test":           "hyperparam_search_v5",
        "best_params":    best_params,
        "top10":          top10,
        "stage1_grid":    {"radius": RADIUS_GRID, "beta": BETA_GRID, "K": K_GRID},
        "stage2_fixed":   {"radius": best_r, "beta": best_b, "K": best_k},
        "n_stage1_combos": n_s1,
        "n_stage2_combos": n_s2,
        "all_results":    sorted(valid_res,
                                 key=lambda x: x["mean_pct_rank"],
                                 reverse=True),
    }

    json_path = os.path.join(RESULTS_DIR, "hyperparam_search_1_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved JSON : {json_path}")

    # Plain-text report
    report_lines = [
        "V5 HYPERPARAMETER SEARCH REPORT — Harris Teeter, Charlotte NC",
        "=" * 65,
        "",
        "BEST COMBINATION",
        f"  Catchment radius : {best_r} miles",
        f"  Beta (decay)     : {best_b}",
        f"  K (competitors)  : {best_k}",
        f"  W1 (gap score)   : {best_w1}",
        f"  W2 (income)      : {best_w2}",
        f"  W3 (access)      : {best_w3}",
        f"  Mean HT pctile   : {best_s2_score:.2f}th (higher = better)",
        "",
        "INTERPRETATION",
        "  Mean percentile rank measures how high the model scores",
        "  actual Harris Teeter locations vs all scored candidates.",
        "  A value of 75 means the average real HT store falls in",
        "  the top 25% of all candidate scores.",
        "",
        "TOP-10 CONFIGURATIONS",
        f"  {'Rank':<5} {'Radius':>7} {'Beta':>6} {'K':>3} "
        f"{'W1':>5} {'W2':>5} {'W3':>5} {'MeanPct':>9}",
        "  " + "-" * 55,
    ]
    for i, r in enumerate(top10, 1):
        report_lines.append(
            f"  {i:<5} {r['radius']:>7.1f} {r['beta']:>6.2f} "
            f"{r['K']:>3} {r['W1']:>5.2f} {r['W2']:>5.2f} "
            f"{r['W3']:>5.2f} {r['mean_pct_rank']:>9.2f}"
        )

    txt_path = os.path.join(RESULTS_DIR, "hyperparam_search_1_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved report: {txt_path}")
    print(f"\n  Use these values in the app sliders for best predictions.")


if __name__ == "__main__":
    main()

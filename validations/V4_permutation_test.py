# =============================================================
# V4_permutation_test.py
#
# Permutation test for V2 calibration validation.
# Uses model_approach2's own scoring function (not reimplemented).
#
# USAGE:   python V4_permutation_test.py
# OUTPUT:  Results_final/V4_results/permutation_test_results.json
# =============================================================

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_approach2 import load_all, score_all_candidates_like_ht, MILE_M, CRS_M

# =============================================================
N_PERMUTATIONS  = 10000
RANDOM_SEED     = 42
RESULTS_DIR     = "Results_final/V4_results"
V5_PARAMS_FILE  = "Results_final/V5_results/optimal_params.json"
V2_PARAMS_FILE  = "Results_final/V2_results/learned_weights.json"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("  V4: PERMUTATION TEST FOR CALIBRATION VALIDATION")
    print("=" * 65)

    # ---- Load params: V5 LOOCV-validated preferred, V2 fallback ----
    if os.path.exists(V5_PARAMS_FILE):
        with open(V5_PARAMS_FILE) as f:
            p = json.load(f)
        print(f"\n  Params source: V5 LOOCV-validated")
    elif os.path.exists(V2_PARAMS_FILE):
        with open(V2_PARAMS_FILE) as f:
            lw = json.load(f)
        p = {"radius_miles": 5.0, "beta": float(lw.get("beta", 3.0)),
             "K": 3, "W1": float(lw["W1"]),
             "W2": float(lw["W2"]), "W3": float(lw["W3"])}
        print(f"\n  Params source: V2 calibrated weights")
    else:
        p = {"radius_miles": 5.0, "beta": 3.0, "K": 3,
             "W1": 0.984, "W2": 0.016, "W3": 0.0003}
        print(f"\n  Params source: hardcoded V2 defaults")

    W1_CAL     = p["W1"]
    W2_CAL     = p["W2"]
    W3_CAL     = p["W3"]
    BETA_CAL   = p["beta"]
    RADIUS_CAL = p["radius_miles"]
    K_CAL      = p.get("K", 3)

    print(f"  radius={RADIUS_CAL}mi  beta={BETA_CAL}  K={K_CAL}")
    print(f"  W1={W1_CAL}  W2={W2_CAL}  W3={W3_CAL}")

    # ---- Step 1: Load data ----
    print("\n[1/4] Loading data...")
    t0 = time.perf_counter()
    state = load_all()
    ht_m    = state["ht_m"]
    cands_m = state["cands_m"]
    N_HT    = len(ht_m)
    N_CANDS = len(cands_m)
    print(f"    BGs: {len(state['bg_m'])}, Candidates: {N_CANDS}, HT: {N_HT}")
    print(f"    Loaded in {time.perf_counter()-t0:.1f}s")

    # ---- Step 2: Score using model's own function ----
    print(f"\n[2/4] Scoring with params "
          f"(W1={W1_CAL}, W2={W2_CAL}, W3={W3_CAL}, "
          f"beta={BETA_CAL}, radius={RADIUS_CAL}mi, K={K_CAL})...")
    t1 = time.perf_counter()

    top_n, heat_points, ht_gdf, out_ll = score_all_candidates_like_ht(
        state,
        W1=W1_CAL, W2=W2_CAL, W3=W3_CAL,
        beta=BETA_CAL, radius_miles=RADIUS_CAL, K=K_CAL,
        return_all=True,
    )

    scores = out_ll["pair_score"].values
    valid_mask = np.isfinite(scores)
    n_valid = valid_mask.sum()
    print(f"    Valid candidates: {n_valid}")
    print(f"    Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")
    print(f"    Scoring done in {time.perf_counter()-t1:.1f}s")

    # Percentile ranks (higher score = higher percentile)
    valid_scores = scores[valid_mask]
    rank_order = np.argsort(np.argsort(valid_scores)).astype(float)
    pct_ranks = (rank_order / (n_valid - 1)) * 100.0

    pct_full = np.full(N_CANDS, np.nan)
    pct_full[valid_mask] = pct_ranks

    # ---- Step 3: Match HT stores to candidates ----
    print(f"\n[3/4] Matching HT stores to candidates...")
    from scipy.spatial import cKDTree

    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    ht_xy   = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    cand_tree = cKDTree(cand_xy)
    ht_dists, ht_idxs = cand_tree.query(ht_xy)

    max_match = 0.5 * MILE_M
    close = ht_dists <= max_match
    ht_matched = ht_idxs[close]
    n_matched = close.sum()
    print(f"    Matched: {n_matched} / {N_HT} (within 0.5 miles)")

    ht_pcts = pct_full[ht_matched]
    ht_valid = ht_pcts[np.isfinite(ht_pcts)]
    observed_mean = np.mean(ht_valid)

    # Also compute mean rank (as in V2 output)
    ht_scores = scores[ht_matched]
    ht_ranks = []
    for s in ht_scores:
        if np.isfinite(s):
            rank = np.sum(valid_scores <= s)  # how many score <= this
            ht_ranks.append(rank)
    mean_rank = np.mean(ht_ranks)
    mean_rank_pct = (mean_rank / n_valid) * 100

    print(f"    Mean HT rank: {mean_rank:.0f} / {n_valid} "
          f"({mean_rank_pct:.1f}th percentile)")
    print(f"    Observed mean percentile: {observed_mean:.1f}")

    # Print individual HT store ranks
    print(f"\n    Individual HT store percentiles:")
    for idx, (d, pct) in enumerate(zip(ht_dists[close], ht_valid)):
        print(f"      Store {idx+1:2d}: {pct:.1f}th pctile "
              f"(match dist: {d:.0f}m)")

    # ---- Step 4: Permutation test ----
    print(f"\n[4/4] Permutation test ({N_PERMUTATIONS:,} iterations)...")
    t2 = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)
    valid_idx = np.where(valid_mask)[0]
    n_ht = len(ht_valid)

    perm_means = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        sample = rng.choice(valid_idx, size=n_ht, replace=False)
        perm_means[p] = np.mean(pct_full[sample])
        if (p + 1) % 2000 == 0:
            print(f"    {p+1:,} / {N_PERMUTATIONS:,}")

    perm_time = time.perf_counter() - t2

    n_exceed = np.sum(perm_means >= observed_mean)
    p_value = n_exceed / N_PERMUTATIONS

    # ---- Results ----
    print(f"\n{'=' * 65}")
    print(f"  PERMUTATION TEST RESULTS")
    print(f"{'=' * 65}")
    print(f"  HT stores matched:          {n_matched}")
    print(f"  Valid scores:                {len(ht_valid)}")
    print(f"  Mean HT rank:               {mean_rank:.0f} / {n_valid} "
          f"({mean_rank_pct:.1f}th pctile)")
    print(f"  Observed mean percentile:    {observed_mean:.1f}")
    print(f"  Permutation mean:            {np.mean(perm_means):.1f}")
    print(f"  Permutation std:             {np.std(perm_means):.1f}")
    print(f"  Random samples >= observed:  {n_exceed} / {N_PERMUTATIONS:,}")
    if p_value < 0.001:
        print(f"  p-value:                     < 0.001")
    else:
        print(f"  p-value:                     {p_value:.4f}")
    print(f"  Significant (p < 0.05):      {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  Test time:                   {perm_time:.1f}s")
    print(f"{'=' * 65}")

    print(f"\n  Stores in top 25%: {np.sum(ht_valid >= 75)}/{len(ht_valid)}")
    print(f"  Stores in top 50%: {np.sum(ht_valid >= 50)}/{len(ht_valid)}")

    # Save
    results = {
        "test": "permutation_test_v2",
        "n_permutations": N_PERMUTATIONS,
        "n_ht_matched": int(n_matched),
        "n_ht_valid": int(len(ht_valid)),
        "n_candidates_valid": int(n_valid),
        "mean_ht_rank": round(float(mean_rank), 0),
        "mean_ht_rank_percentile": round(float(mean_rank_pct), 1),
        "observed_mean_percentile": round(float(observed_mean), 1),
        "permutation_mean": round(float(np.mean(perm_means)), 1),
        "permutation_std": round(float(np.std(perm_means)), 1),
        "n_exceed": int(n_exceed),
        "p_value": round(float(p_value), 4),
        "p_value_str": "< 0.001" if p_value < 0.001 else f"{p_value:.4f}",
        "significant": bool(p_value < 0.05),
        "ht_percentiles": [round(float(p), 1) for p in ht_valid],
    }
    out = os.path.join(RESULTS_DIR, "permutation_test_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

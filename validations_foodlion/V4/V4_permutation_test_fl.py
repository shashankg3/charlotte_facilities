# =============================================================
# V4_permutation_test_fl.py  (Food Lion study)
#
# Replication of validations/V4_permutation_test.py for Food Lion.
# Uses calibrated weights from V2 Food Lion output.
#
# Workflow:
#   1. Load V2 learned weights from Results_final/FL_results/V2/learned_weights.json
#   2. Score all 2,450 candidates with those weights
#   3. Match 27 real Food Lion stores to nearest candidates
#   4. Run 10,000-draw permutation test
#   5. Compute p-value and save results
#
# OUTPUT: Results_final/FL_results/V4/permutation_test_results.json
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations_foodlion/V4/V4_permutation_test_fl.py
#
# NOTE: Run V2 first so learned_weights.json exists.
# =============================================================

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_foodlion import load_all, score_all_candidates_like_ht, MILE_M

# =============================================================
N_PERMUTATIONS  = 10_000
RANDOM_SEED     = 42
RESULTS_DIR     = "Results_final/FL_results/V4"
PARAMS_FILE     = "Results_final/FL_results/V4/optimal_params.json"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  V4: PERMUTATION TEST — Food Lion")
    print("=" * 65)

    # ---- Load LOOCV-validated params from V5 Search 3 ----
    if not os.path.exists(PARAMS_FILE):
        print(f"[V4-FL] ERROR: params file not found: {PARAMS_FILE}")
        sys.exit(1)

    with open(PARAMS_FILE) as f:
        lw = json.load(f)

    W1_CAL     = float(lw["W1"])
    W2_CAL     = float(lw["W2"])
    W3_CAL     = float(lw["W3"])
    BETA_CAL   = float(lw["beta"])
    RADIUS_CAL = float(lw["radius_miles"])
    K_CAL      = int(lw["K"])

    print(f"\n[1/4] Loaded LOOCV-validated params (V5 Search 3):")
    print(f"      radius={RADIUS_CAL}mi, beta={BETA_CAL}, K={K_CAL}")
    print(f"      W1={W1_CAL:.4f}, W2={W2_CAL:.4f}, W3={W3_CAL:.4f}")
    print(f"      LOOCV pct rank: {lw.get('loocv_pct_rank','?')}th")

    # ---- Load model state ----
    print(f"\n[2/4] Loading Food Lion model state...")
    t0 = time.perf_counter()
    state = load_all()
    fl_m    = state["ht_m"]    # Food Lion stores (named ht_m internally)
    cands_m = state["cands_m"]
    N_FL    = len(fl_m)
    N_CANDS = len(cands_m)
    print(f"      BGs: {len(state['bg_m'])}, "
          f"Candidates: {N_CANDS}, "
          f"Food Lion stores: {N_FL}")
    print(f"      Loaded in {time.perf_counter()-t0:.1f}s")

    # ---- Score all candidates with calibrated weights ----
    print(f"\n[3/4] Scoring with calibrated weights...")
    t1 = time.perf_counter()

    top_n, heat_points, fl_gdf, out_ll = score_all_candidates_like_ht(
        state,
        W1=W1_CAL, W2=W2_CAL, W3=W3_CAL,
        beta=BETA_CAL, radius_miles=RADIUS_CAL, K=K_CAL,
        return_all=True,
    )

    scores     = out_ll["pair_score"].values
    valid_mask = np.isfinite(scores)
    n_valid    = valid_mask.sum()
    print(f"      Valid candidates: {n_valid}")
    print(f"      Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")
    print(f"      Scoring done in {time.perf_counter()-t1:.1f}s")

    valid_scores = scores[valid_mask]
    rank_order   = np.argsort(np.argsort(valid_scores)).astype(float)
    pct_ranks    = (rank_order / (n_valid - 1)) * 100.0

    pct_full = np.full(N_CANDS, np.nan)
    pct_full[valid_mask] = pct_ranks

    # ---- Match FL stores to nearest candidates ----
    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    fl_xy   = np.c_[fl_m.geometry.x.values,   fl_m.geometry.y.values]
    cand_tree          = cKDTree(cand_xy)
    fl_dists, fl_idxs  = cand_tree.query(fl_xy)

    max_match  = 0.5 * MILE_M
    close      = fl_dists <= max_match
    fl_matched = fl_idxs[close]
    n_matched  = close.sum()
    print(f"\n      FL stores matched (within 0.5 mi): {n_matched} / {N_FL}")

    fl_pcts    = pct_full[fl_matched]
    fl_valid   = fl_pcts[np.isfinite(fl_pcts)]
    observed_mean = np.mean(fl_valid)

    fl_scores_matched = scores[fl_matched]
    fl_ranks = [int(np.sum(valid_scores <= s))
                for s in fl_scores_matched if np.isfinite(s)]
    mean_rank     = np.mean(fl_ranks)
    mean_rank_pct = (mean_rank / n_valid) * 100

    print(f"      Mean FL rank: {mean_rank:.0f} / {n_valid} "
          f"({mean_rank_pct:.1f}th percentile)")
    print(f"      Observed mean percentile: {observed_mean:.1f}")

    print(f"\n      Individual FL store percentiles:")
    for idx, (d, pct) in enumerate(zip(fl_dists[close], fl_valid)):
        print(f"        Store {idx+1:2d}: {pct:.1f}th pctile "
              f"(match dist: {d:.0f}m)")

    # ---- Permutation test ----
    print(f"\n[4/4] Permutation test ({N_PERMUTATIONS:,} draws)...")
    t2   = time.perf_counter()
    rng  = np.random.default_rng(RANDOM_SEED)
    valid_idx = np.where(valid_mask)[0]
    n_fl_v    = len(fl_valid)

    perm_means = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        sample        = rng.choice(valid_idx, size=n_fl_v, replace=False)
        perm_means[p] = np.mean(pct_full[sample])
        if (p + 1) % 2000 == 0:
            print(f"      {p+1:,} / {N_PERMUTATIONS:,} done...")

    perm_time = time.perf_counter() - t2
    n_exceed  = int(np.sum(perm_means >= observed_mean))
    p_value   = n_exceed / N_PERMUTATIONS

    # ---- Results ----
    print(f"\n{'=' * 65}")
    print(f"  PERMUTATION TEST RESULTS — Food Lion")
    print(f"{'=' * 65}")
    print(f"  FL stores matched:           {n_matched}")
    print(f"  Valid scores:                {len(fl_valid)}")
    print(f"  Mean FL rank:                {mean_rank:.0f} / {n_valid} "
          f"({mean_rank_pct:.1f}th pctile)")
    print(f"  Observed mean percentile:    {observed_mean:.1f}")
    print(f"  Permutation mean:            {np.mean(perm_means):.1f}")
    print(f"  Permutation std:             {np.std(perm_means):.1f}")
    print(f"  Random samples >= observed:  {n_exceed} / {N_PERMUTATIONS:,}")
    print(f"  p-value:                     "
          f"{'< 0.001' if p_value < 0.001 else f'{p_value:.4f}'}")
    print(f"  Significant (p < 0.05):      {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  Test time:                   {perm_time:.1f}s")
    print(f"{'=' * 65}")

    results = {
        "brand":                     "Food Lion",
        "city":                      "Charlotte, NC",
        "test":                      "permutation_test_v4_foodlion",
        "params_source":             "V5_hyperparam_search_3_loocv",
        "optimal_params":            lw,
        "loocv_pct_rank":            lw.get("loocv_pct_rank"),
        "n_permutations":            N_PERMUTATIONS,
        "n_fl_matched":              int(n_matched),
        "n_fl_valid":                int(len(fl_valid)),
        "n_candidates_valid":        int(n_valid),
        "mean_fl_rank":              round(float(mean_rank), 0),
        "mean_fl_rank_percentile":   round(float(mean_rank_pct), 1),
        "observed_mean_percentile":  round(float(observed_mean), 1),
        "permutation_mean":          round(float(np.mean(perm_means)), 1),
        "permutation_std":           round(float(np.std(perm_means)), 1),
        "n_exceed":                  n_exceed,
        "p_value":                   round(float(p_value), 4),
        "p_value_str":               "< 0.001" if p_value < 0.001 else f"{p_value:.4f}",
        "significant":               bool(p_value < 0.05),
        "fl_percentiles":            [round(float(p), 1) for p in fl_valid],
    }

    out = os.path.join(RESULTS_DIR, "permutation_test_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

# =============================================================
# weight_calibration.py
#
# Weight justification for the facility location model.
#
# OPTION 1 -- Data-driven weight learning
#   Uses real Harris Teeter store locations as ground truth.
#   Finds W1, W2, W3, lambda, beta that minimise the average
#   rank of real HT stores in the scored candidate pool.
#   This makes the weight choice defensible and publishable.
#
# OPTION 2 -- Sensitivity analysis
#   Sweeps a grid of weight combinations and reports which
#   top-10 candidate sites appear consistently across configs.
#   Robust sites (appearing in >=70% of configs) are reported.
#
# USAGE:
#   python weight_calibration.py
#
# OUTPUT:
#   results/calibration_report.txt   -- full text report
#   results/learned_weights.json     -- best weights (use in app)
#   results/sensitivity_summary.csv  -- per-candidate stability
#   results/sensitivity_heatmap.csv  -- weight grid results
# =============================================================

import os
import sys
import json
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.optimize import minimize, differential_evolution
from scipy.sparse.csgraph import dijkstra as csr_dijkstra

warnings.filterwarnings("ignore")

# Add project root to path so we can import model_approach2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_approach2 import (
    load_all,
    norm_weight,
    huff_share_vs_competitors,
    _nearest_node_idx,
    _first_existing,
    MILE_M,
    CRS_M,
    CRS_LL,
    CIRCUITY_FACTOR,
    ACCESS_ALPHA,
)

os.makedirs("results", exist_ok=True)

SPEED_MPS = 35 * MILE_M / 3600.0


# =============================================================
# FAST SCORER
# Pre-computes everything that doesn't depend on weights,
# then scoring a new weight combo is just arithmetic -- no
# re-reading data, no re-running Dijkstra.
# =============================================================

class FastScorer:
    """
    Pre-computes all expensive operations (Dijkstra, Huff shares,
    KDTree lookups) once.  Subsequent calls with different weights
    are pure NumPy arithmetic -- takes ~1ms per call.

    This makes the optimisation feasible (hundreds of evaluations).
    """

    def __init__(self, state, radius_miles=5.0, K=3):
        print("[FastScorer] Pre-computing score components...")
        t0 = time.perf_counter()

        self.radius_miles = radius_miles
        rad_m  = radius_miles * MILE_M
        cutoff = rad_m * CIRCUITY_FACTOR

        ht_m        = state["ht_m"]
        comp_m      = state["comp_m"]
        cands_m     = state["cands_m"]
        cands_ll    = state["cands_ll"].copy().reset_index(drop=True)
        bg_m        = state["bg_m"]
        road_csr       = state.get("road_csr",        None)
        road_kdtree    = state.get("road_kdtree",    None)
        bg_node_ids    = state.get("bg_node_ids",    None)
        bg_dist_matrix = state.get("bg_dist_matrix", None)  # (n_bgs x n_nodes), precomputed

        # ---- BG attributes ----
        bg_cent = bg_m.copy()
        if "cent" not in bg_cent.columns:
            bg_cent["cent"] = bg_cent.geometry.centroid
        bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

        pop_col = _first_existing(bg_cent.columns,
                                  ["population","pop_total","pop",
                                   "POP","B01001_001E"])
        if pop_col is None:
            raise KeyError("No population column in bg_m.")
        bg_pop = pd.to_numeric(bg_cent[pop_col], errors="coerce").fillna(0).to_numpy()

        inc_col = _first_existing(bg_cent.columns,
                                  ["income","median_income",
                                   "med_income","B19013_001E"])
        bg_inc  = (pd.to_numeric(bg_cent[inc_col], errors="coerce").fillna(0).to_numpy()
                   if inc_col else np.zeros_like(bg_pop))

        dens_col = _first_existing(bg_cent.columns,
                                   ["pop_per_sqmi","density","DENSITY"])
        dens_w   = (norm_weight(pd.to_numeric(bg_cent[dens_col],
                                errors="coerce").fillna(0).to_numpy())
                    if dens_col else np.ones_like(bg_pop))

        block_weight = (0.4 * norm_weight(bg_pop)
                      + 0.3 * norm_weight(bg_inc)
                      + 0.3 * dens_w)

        bg_tree  = cKDTree(bg_xy)

        # ---- Competitor times from each BG (for Huff) ----
        if not comp_m.empty:
            comp_xy   = np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values]
            comp_tree = cKDTree(comp_xy)
            k_eff     = min(max(1, K), len(comp_xy))
            d, _      = comp_tree.query(bg_xy, k=k_eff)
            if d.ndim == 1:
                d = d[:, None]
            bg_tcomp  = (d / SPEED_MPS) / 60.0
        else:
            bg_tcomp = np.full((len(bg_xy), 1), 60.0)

        # ---- All stores for saturation ----
        xs = np.concatenate([ht_m.geometry.x.values, comp_m.geometry.x.values])
        ys = np.concatenate([ht_m.geometry.y.values, comp_m.geometry.y.values])
        all_store_tree = cKDTree(np.c_[xs, ys]) if len(xs) > 0 else None

        # ---- Candidate coordinates ----
        cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
        N_cands = len(cand_xy)

        # ---- Pre-compute per-candidate arrays ----
        # These are INDEPENDENT of weights -- computed once
        print(f"[FastScorer] Computing components for {N_cands} candidates...")

        pop_arr       = np.zeros(N_cands)
        s10k_arr      = np.zeros(N_cands)   # competition (all stores / 10k pop)
        access_arr    = np.zeros(N_cands)   # Dijkstra accessibility
        potential_arr = np.zeros(N_cands)   # Huff potential (beta=2.0 default)

        # We store raw Huff numerator and denominator separately so we
        # can re-compute potential for any beta without re-running KDTree
        # NOTE: we pre-compute at beta=2.0; for different betas we recompute
        # (still fast since it's just array ops on pre-queried distances)
        self._bg_xy       = bg_xy
        self._bg_tree     = bg_tree
        self._bg_pop      = bg_pop
        self._block_weight = block_weight
        self._bg_tcomp    = bg_tcomp
        self._cand_xy     = cand_xy
        self._all_store_tree = all_store_tree
        self._road_csr    = road_csr
        self._road_kdtree = road_kdtree
        self._bg_node_ids = bg_node_ids
        self._rad_m       = rad_m
        self._cutoff      = cutoff
        self._cands_ll    = cands_ll

        # Pre-compute BG indices and distances for each candidate
        self._cand_bg_idxs  = []   # list of arrays
        self._cand_bg_tnew  = []   # list of arrays (travel time to candidate)
        self._cand_node_ids = []   # road node per candidate

        for i, (cx, cy) in enumerate(cand_xy):
            # Snap candidate to road node first (needed for road distance lookup)
            cand_node = (_nearest_node_idx(road_kdtree, cx, cy)
                         if road_kdtree is not None else None)
            self._cand_node_ids.append(cand_node)

            idxs = bg_tree.query_ball_point([cx, cy], r=rad_m)
            if idxs:
                idxs_arr = np.asarray(idxs, dtype=np.int32)
                pop_buf  = float(np.sum(bg_pop[idxs_arr]))
                pop_arr[i] = pop_buf

                # FIX 1+2: use road distances from precomputed bg_dist_matrix
                # bg_dist_matrix[j, cand_node] = road dist from BG j to candidate
                if bg_dist_matrix is not None and cand_node is not None:
                    dist_m = bg_dist_matrix[idxs_arr, cand_node].astype(np.float64)
                    # unreachable nodes get cutoff distance (not inf)
                    dist_m = np.where(np.isfinite(dist_m), dist_m,
                                      rad_m * CIRCUITY_FACTOR)
                    t_new  = (dist_m / SPEED_MPS) / 60.0
                else:
                    # fallback: Euclidean (only if no precomputed matrix)
                    d_new = np.hypot(bg_xy[idxs_arr, 0] - cx,
                                     bg_xy[idxs_arr, 1] - cy)
                    t_new = (d_new / SPEED_MPS) / 60.0

                if all_store_tree is not None and pop_buf > 0:
                    ct = len(all_store_tree.query_ball_point([cx, cy], r=rad_m))
                    s10k_arr[i] = ct / (pop_buf / 10_000.0)
            else:
                idxs_arr = np.array([], dtype=np.int32)
                t_new    = np.array([], dtype=np.float64)

            self._cand_bg_idxs.append(idxs_arr)
            self._cand_bg_tnew.append(t_new)

        self._pop_arr  = pop_arr
        self._s10k_arr = s10k_arr

        # Compute accessibility from precomputed road distances (no Dijkstra!)
        # _cand_bg_tnew already holds road travel times from bg_dist_matrix
        print("[FastScorer] Computing accessibility from precomputed road distances...")
        t_acc = time.perf_counter()
        for i in range(N_cands):
            idxs_arr   = self._cand_bg_idxs[i]
            t_road_min = self._cand_bg_tnew[i]
            if len(idxs_arr) == 0:
                continue
            w_loc = block_weight[idxs_arr]
            valid = np.isfinite(t_road_min) & (t_road_min > 0) & (w_loc > 0)
            if valid.any():
                t_bar = float(np.nansum(w_loc[valid] * t_road_min[valid])
                              / np.nansum(w_loc[valid]))
                access_arr[i] = float(np.clip(
                    np.exp(-ACCESS_ALPHA * t_bar), 0.01, 1.0))

        print(f"[FastScorer] Accessibility done in {time.perf_counter()-t_acc:.2f}s  "
              f"(no Dijkstra -- used precomputed bg_dist_matrix)")
        self._access_arr = access_arr

        # Store candidate geometries for output
        self._cands_ll = cands_ll

        elapsed = time.perf_counter() - t0
        print(f"[FastScorer] Pre-computation done in {elapsed:.1f}s")

    def score(self, W1, W2, W3, beta):
        """
        Score all candidates with given parameters.
        Formula: Score_k = W1*P~_k + W2*A~_k - W3*S~_k
        All three components normalised to [0.01, 1.0] before combining.
        lambda removed -- W3 controls competition strength directly.
        """
        N = len(self._cand_xy)
        potential_arr = np.zeros(N)

        for i in range(N):
            idxs_arr = self._cand_bg_idxs[i]
            t_new    = self._cand_bg_tnew[i]
            if len(idxs_arr) == 0:
                continue
            share = huff_share_vs_competitors(
                t_new, self._bg_tcomp[idxs_arr, :], beta)
            potential_arr[i] = float(
                np.nansum(self._block_weight[idxs_arr] * share))

        # Normalise all three to [0.01, 1.0] -- ensures weights are meaningful
        potential_norm = norm_weight(potential_arr)
        access_norm    = norm_weight(self._access_arr)
        s10k_norm      = norm_weight(self._s10k_arr)

        scores = W1 * potential_norm + W2 * access_norm - W3 * s10k_norm
        return scores, potential_norm

    def rank_of_locations(self, query_xy_m, scores):
        """
        For each query location (real HT store in metric CRS),
        find the nearest candidate and return its rank in the
        scored pool (1 = best).

        Returns list of (rank, score, distance_m) tuples.
        """
        cand_tree = cKDTree(self._cand_xy)
        ranks     = []
        score_order = np.argsort(-scores)  # descending
        rank_lookup = np.empty_like(score_order)
        rank_lookup[score_order] = np.arange(1, len(scores) + 1)

        for (qx, qy) in query_xy_m:
            d, nearest = cand_tree.query([qx, qy], k=1)
            rank = int(rank_lookup[nearest])
            ranks.append({
                "rank":       rank,
                "score":      float(scores[nearest]),
                "dist_m":     float(d),
                "percentile": float((1.0 - rank / len(scores)) * 100),
            })
        return ranks


# =============================================================
# OPTION 1 -- DATA-DRIVEN WEIGHT LEARNING
# =============================================================

def learn_weights(state, radius_miles=5.0, K=3, seed=42):
    """
    Learn W1, W2, W3, lambda, beta by minimising the average rank
    of real Harris Teeter stores in the candidate scoring pool.

    Objective: minimise mean rank of real HT locations.
    Lower rank = model scores real HT sites highly = model agrees
    with real-world retail decisions.

    Uses two-stage optimisation:
      Stage 1: Differential Evolution (global, robust to local minima)
      Stage 2: Nelder-Mead (local refinement from DE result)
    """
    print("\n" + "="*60)
    print("OPTION 1: DATA-DRIVEN WEIGHT LEARNING")
    print("="*60)

    scorer    = FastScorer(state, radius_miles=radius_miles, K=K)
    ht_m      = state["ht_m"].reset_index(drop=True)
    ht_xy_m   = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]

    n_ht      = len(ht_m)
    n_cands   = len(scorer._cand_xy)
    print(f"\nGround truth: {n_ht} real HT stores")
    print(f"Candidate pool: {n_cands} locations")

    eval_count = [0]

    def objective(params):
        W1, W2, W3, beta = params

        if W1 <= 0 or W2 <= 0 or W3 <= 0 or beta <= 0:
            return 1e9
        wsum = W1 + W2 + W3
        W1n, W2n, W3n = W1/wsum, W2/wsum, W3/wsum

        scores, _ = scorer.score(W1n, W2n, W3n, beta)
        results   = scorer.rank_of_locations(ht_xy_m, scores)
        ranks     = [r["rank"] for r in results]
        mean_rank = float(np.mean(ranks))

        eval_count[0] += 1
        if eval_count[0] % 50 == 0:
            print(f"  eval {eval_count[0]:4d}: "
                  f"W=({W1n:.2f},{W2n:.2f},{W3n:.2f}) "
                  f"beta={beta:.2f} "
                  f"-> mean_rank={mean_rank:.1f} / {n_cands}")
        return mean_rank

    # Parameter bounds: [W1, W2, W3, beta]
    bounds = [(0.1, 1.0),   # W1
              (0.1, 1.0),   # W2
              (0.1, 1.0),   # W3
              (1.0, 3.0)]   # beta

    print("\n--- Stage 1: Differential Evolution (global search) ---")
    t0 = time.perf_counter()
    de_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=200,
        popsize=12,
        tol=0.01,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        polish=False,
    )
    print(f"DE done in {time.perf_counter()-t0:.1f}s  "
          f"best={de_result.fun:.2f}  success={de_result.success}")

    print("\n--- Stage 2: Nelder-Mead refinement ---")
    nm_result = minimize(
        objective,
        x0=de_result.x,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 0.1},
    )
    print(f"NM done  best={nm_result.fun:.2f}  success={nm_result.success}")

    # ---- Extract best parameters ----
    W1r, W2r, W3r, beta_r = nm_result.x
    wsum = W1r + W2r + W3r
    W1f, W2f, W3f = W1r/wsum, W2r/wsum, W3r/wsum
    beta_f = float(np.clip(beta_r, 1.0, 3.0))

    # ---- Evaluate with learned weights ----
    scores_learned, pot_norm = scorer.score(W1f, W2f, W3f, beta_f)
    results_learned = scorer.rank_of_locations(ht_xy_m, scores_learned)
    ranks_learned   = [r["rank"]       for r in results_learned]
    pcts_learned    = [r["percentile"] for r in results_learned]

    # ---- Baseline: equal weights ----
    scores_equal, _ = scorer.score(1/3, 1/3, 1/3, 2.0)
    results_equal   = scorer.rank_of_locations(ht_xy_m, scores_equal)
    ranks_equal     = [r["rank"] for r in results_equal]

    # ---- Baseline: paper defaults ----
    scores_paper, _ = scorer.score(0.4, 0.3, 0.3, 2.0)
    results_paper   = scorer.rank_of_locations(ht_xy_m, scores_paper)
    ranks_paper     = [r["rank"] for r in results_paper]

    report = {
        "learned_weights": {
            "W1":    round(W1f,   4),
            "W2":    round(W2f,   4),
            "W3":    round(W3f,   4),
            "beta":  round(beta_f, 4),
        },
        "optimisation": {
            "objective_value":  round(float(nm_result.fun), 2),
            "n_evaluations":    int(eval_count[0]),
            "de_success":       bool(de_result.success),
            "nm_success":       bool(nm_result.success),
        },
        "ht_rank_stats": {
            "learned": {
                "mean_rank":       round(float(np.mean(ranks_learned)), 2),
                "median_rank":     round(float(np.median(ranks_learned)), 2),
                "mean_percentile": round(float(np.mean(pcts_learned)), 2),
                "pct_top10pct":    round(float(np.mean(
                                    np.array(pcts_learned) >= 90)) * 100, 1),
                "pct_top25pct":    round(float(np.mean(
                                    np.array(pcts_learned) >= 75)) * 100, 1),
            },
            "equal_weights": {
                "mean_rank":   round(float(np.mean(ranks_equal)), 2),
                "median_rank": round(float(np.median(ranks_equal)), 2),
            },
            "paper_defaults": {
                "mean_rank":   round(float(np.mean(ranks_paper)), 2),
                "median_rank": round(float(np.median(ranks_paper)), 2),
            },
        },
        "per_ht_store":    results_learned,
        "n_ht_stores":     n_ht,
        "n_candidates":    n_cands,
        "radius_miles":    radius_miles,
    }

    return report, scorer, scores_learned


# =============================================================
# OPTION 2 -- SENSITIVITY ANALYSIS
# =============================================================

def sensitivity_analysis(scorer, learned_weights, n_top=10, min_sep_miles=2.5):
    """
    Sweep a grid of weight combinations around the learned weights.
    Report which top-N candidate sites appear consistently across
    all weight configurations.

    A site that appears in >=70% of configurations is called 'robust'.
    """
    print("\n" + "="*60)
    print("OPTION 2: SENSITIVITY ANALYSIS")
    print("="*60)

    lw = learned_weights

    # Grid around learned weights ±50%
    W1_vals   = np.clip(
        [lw["W1"] * f for f in [0.5, 0.7, 1.0, 1.3, 1.5]], 0.05, 0.90)
    W2_vals   = np.clip(
        [lw["W2"] * f for f in [0.5, 0.7, 1.0, 1.3, 1.5]], 0.05, 0.90)
    beta_vals = [max(1.0, lw["beta"] + d) for d in [-0.5, -0.25, 0.0, 0.25, 0.5]]

    configs = []
    for W1, W2, beta in itertools.product(W1_vals, W2_vals, beta_vals):
        W3   = max(0.05, 1.0 - W1 - W2)
        wsum = W1 + W2 + W3
        configs.append((W1/wsum, W2/wsum, W3/wsum, beta))

    print(f"\nTotal weight configurations: {len(configs)}")
    print("Running scoring for each configuration...")

    min_sep_m = min_sep_miles * MILE_M
    cand_xy   = scorer._cand_xy
    n_cands   = len(cand_xy)

    # Count how many times each candidate appears in top-N
    appearance_count = np.zeros(n_cands, dtype=np.int32)
    grid_results     = []

    for i, (W1, W2, W3, beta) in enumerate(configs):
        scores, _ = scorer.score(W1, W2, W3, beta)

        # Greedy diverse top-N selection
        order    = np.argsort(-scores)
        selected = []
        for idx in order:
            if len(selected) >= n_top:
                break
            if not selected:
                selected.append(idx)
                continue
            dx   = cand_xy[idx, 0] - cand_xy[selected, 0]
            dy   = cand_xy[idx, 1] - cand_xy[selected, 1]
            dist = np.hypot(dx, dy)
            if np.all(dist >= min_sep_m):
                selected.append(idx)

        for idx in selected:
            appearance_count[idx] += 1

        grid_results.append({
            "config_id": i,
            "W1": round(W1, 3), "W2": round(W2, 3), "W3": round(W3, 3),
            "beta": round(beta, 3),
            "top_candidates": [int(x) for x in selected],
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(configs)} configs done...")

    # ---- Stability report ----
    stability = appearance_count / len(configs)   # fraction of configs
    top_stable_idx = np.argsort(-stability)[:20]  # top 20 most stable

    cand_ll    = scorer._cands_ll
    cands_m_xy = scorer._cand_xy

    stable_sites = []
    for idx in top_stable_idx:
        row = cand_ll.iloc[int(idx)]
        geom = row.geometry
        stable_sites.append({
            "candidate_idx":     int(idx),
            "lat":               float(geom.y),
            "lon":               float(geom.x),
            "appearance_frac":   round(float(stability[idx]), 4),
            "appearance_pct":    round(float(stability[idx]) * 100, 1),
            "is_robust":         bool(stability[idx] >= 0.70),
            "n_configs_appeared": int(appearance_count[idx]),
            "n_configs_total":   len(configs),
        })

    n_robust = int(np.sum(stability >= 0.70))
    print(f"\nRobust sites (>=70% appearance): {n_robust}")
    print(f"Stable sites (>=50% appearance): {int(np.sum(stability >= 0.50))}")

    # ---- Weight sensitivity on learned config ----
    lw = learned_weights
    sensitivity_dims = {}

    for dim, values, label in [
        ("W1",   W1_vals,   "Market Potential Weight (W1)"),
        ("W2",   W2_vals,   "Accessibility Weight (W2)"),
        ("beta", beta_vals, "Huff Decay β"),
    ]:
        dim_results = []
        for v in values:
            if dim == "W1":
                W1x, W2x, W3x = v, lw["W2"], max(0.05, 1.0 - v - lw["W2"])
            elif dim == "W2":
                W1x, W2x, W3x = lw["W1"], v, max(0.05, 1.0 - lw["W1"] - v)
            else:
                W1x, W2x, W3x = lw["W1"], lw["W2"], lw["W3"]

            beta_v = v if dim == "beta" else lw["beta"]
            wsum   = W1x + W2x + W3x
            scores, _ = scorer.score(
                W1x/wsum, W2x/wsum, W3x/wsum, beta_v)

            order    = np.argsort(-scores)
            selected = []
            for idx in order:
                if len(selected) >= n_top:
                    break
                if not selected:
                    selected.append(idx)
                    continue
                dx   = cand_xy[idx, 0] - cand_xy[selected, 0]
                dy   = cand_xy[idx, 1] - cand_xy[selected, 1]
                dist = np.hypot(dx, dy)
                if np.all(dist >= min_sep_m):
                    selected.append(idx)

            dim_results.append({
                "value":          round(float(v), 3),
                "top_candidates": [int(x) for x in selected],
            })
        sensitivity_dims[dim] = {"label": label, "results": dim_results}

    # Overlap between extreme values (first vs last in each dimension)
    overlap_report = {}
    for dim, data in sensitivity_dims.items():
        res   = data["results"]
        s_low = set(res[0]["top_candidates"])
        s_hi  = set(res[-1]["top_candidates"])
        inter = s_low & s_hi
        overlap_report[dim] = {
            "overlap_count": len(inter),
            "overlap_pct":   round(len(inter) / n_top * 100, 1),
            "stable_sites":  list(inter),
        }

    return {
        "n_configs":       len(configs),
        "n_robust_sites":  n_robust,
        "robust_threshold": 0.70,
        "top_stable_sites": stable_sites,
        "per_dimension_overlap": overlap_report,
        "grid_results":    grid_results,
        "stability_array": stability.tolist(),
    }


# =============================================================
# REPORT WRITER
# =============================================================

def write_report(opt1_report, opt2_report, output_dir="results"):
    lines = []
    lines.append("=" * 65)
    lines.append("  WEIGHT CALIBRATION REPORT")
    lines.append("  Harris Teeter Site Selection -- Charlotte, NC")
    lines.append("=" * 65)

    # --- Option 1 ---
    lines.append("\n--- OPTION 1: DATA-DRIVEN WEIGHT LEARNING ---\n")
    lw = opt1_report["learned_weights"]
    lines.append(f"  Learned weights:")
    lines.append(f"    W1 (market potential) = {lw['W1']:.4f}")
    lines.append(f"    W2 (accessibility)    = {lw['W2']:.4f}")
    lines.append(f"    W3 (competition)      = {lw['W3']:.4f}")
    lines.append(f"    β  (Huff decay)       = {lw['beta']:.4f}")

    lines.append(f"\n  Calibrated against {opt1_report['n_ht_stores']} "
                 f"real Harris Teeter stores")
    lines.append(f"  Candidate pool size: {opt1_report['n_candidates']}")

    rs = opt1_report["ht_rank_stats"]
    lines.append(f"\n  Real HT store rank statistics (lower = better):")
    lines.append(f"  {'Config':<22} {'Mean Rank':>10} {'Median Rank':>12}")
    lines.append(f"  {'-'*46}")
    for cfg, label in [("learned", "Learned weights"),
                       ("paper_defaults", "Paper defaults (0.4/0.3/0.3)"),
                       ("equal_weights",  "Equal weights (1/3 each)")]:
        lines.append(f"  {label:<22} "
                     f"{rs[cfg]['mean_rank']:>10.1f} "
                     f"{rs[cfg]['median_rank']:>12.1f}")

    ld = rs["learned"]
    lines.append(f"\n  With learned weights:")
    lines.append(f"    Mean percentile of real HT stores: "
                 f"{ld['mean_percentile']:.1f}th")
    lines.append(f"    Fraction in top 10%: {ld['pct_top10pct']:.0f}%")
    lines.append(f"    Fraction in top 25%: {ld['pct_top25pct']:.0f}%")

    # --- Option 2 ---
    lines.append("\n\n--- OPTION 2: SENSITIVITY ANALYSIS ---\n")
    lines.append(f"  Weight configurations tested: {opt2_report['n_configs']}")
    lines.append(f"  Robust sites (appear in >=70% of configs): "
                 f"{opt2_report['n_robust_sites']}")

    lines.append(f"\n  Top stable candidate sites:")
    lines.append(f"  {'Rank':<6} {'Lat':>10} {'Lon':>11} "
                 f"{'Appear %':>10} {'Robust':>8}")
    lines.append(f"  {'-'*50}")
    for i, s in enumerate(opt2_report["top_stable_sites"][:10], 1):
        lines.append(f"  {i:<6} {s['lat']:>10.5f} {s['lon']:>11.5f} "
                     f"{s['appearance_pct']:>9.1f}% "
                     f"{'YES' if s['is_robust'] else 'no':>8}")

    lines.append(f"\n  Per-dimension sensitivity (overlap of top-10 "
                 f"between extreme values):")
    for dim, data in opt2_report["per_dimension_overlap"].items():
        lines.append(f"    {dim:<10}: {data['overlap_count']}/{10} sites "
                     f"stable ({data['overlap_pct']:.0f}% overlap)")

    lines.append("\n" + "=" * 65)
    lines.append("  CONCLUSION")
    lines.append("=" * 65)
    lw = opt1_report["learned_weights"]
    mean_pct = opt1_report["ht_rank_stats"]["learned"]["mean_percentile"]
    n_robust = opt2_report["n_robust_sites"]
    lines.append(f"""
  The model was calibrated using {opt1_report['n_ht_stores']} known Harris Teeter
  store locations as ground truth. The learned weights
  (W1={lw['W1']:.2f}, W2={lw['W2']:.2f}, W3={lw['W3']:.2f}, beta={lw['beta']:.2f})
  place real HT stores at the {mean_pct:.0f}th percentile on average
  in the scored candidate pool, indicating the model successfully
  identifies high-scoring regions consistent with real decisions.

  Sensitivity analysis across {opt2_report['n_configs']} weight configurations
  shows {n_robust} candidate sites appear in the top-10 recommendations
  in >=70% of all tested configurations, demonstrating that the
  spatial recommendations are robust to reasonable parameter variation.
""")

    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "calibration_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Report] Written to {path}")
    print(report_text)
    return report_text


# =============================================================
# MAIN
# =============================================================

def main():
    print("Loading model state...")
    state = load_all()
    print(f"State loaded. HT stores: {len(state['ht_m'])}, "
          f"Candidates: {len(state['cands_m'])}")

    RADIUS_MILES = 5.0
    K            = 3

    # ---- Option 1: Learn weights ----
    opt1_report, scorer, scores_learned = learn_weights(
        state,
        radius_miles=RADIUS_MILES,
        K=K,
        seed=42,
    )

    # Save learned weights
    lw_path = os.path.join("results", "learned_weights.json")
    with open(lw_path, "w") as f:
        json.dump(opt1_report["learned_weights"], f, indent=2)
    print(f"\n[Weights] Saved to {lw_path}")
    print(f"  Learned: {opt1_report['learned_weights']}")

    # ---- Option 2: Sensitivity analysis ----
    opt2_report = sensitivity_analysis(
        scorer,
        learned_weights=opt1_report["learned_weights"],
        n_top=10,
        min_sep_miles=2.5,
    )

    # Save sensitivity summary CSV
    stable_df = pd.DataFrame(opt2_report["top_stable_sites"])
    stable_df.to_csv(
        os.path.join("results", "sensitivity_summary.csv"), index=False)

    grid_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "top_candidates"}
        for r in opt2_report["grid_results"]
    ])
    grid_df.to_csv(
        os.path.join("results", "sensitivity_heatmap.csv"), index=False)

    # ---- Write full report ----
    write_report(opt1_report, opt2_report)

    print("\n[Done] All results saved to results/")
    print("  - calibration_report.txt")
    print("  - learned_weights.json      ← use these in app.py")
    print("  - sensitivity_summary.csv")
    print("  - sensitivity_heatmap.csv")

    return opt1_report, opt2_report


if __name__ == "__main__":
    main()
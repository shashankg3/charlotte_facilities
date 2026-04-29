# =============================================================
# V2_weight_calibration_fl.py  (Food Lion study)
#
# Replication of validations/V2_weight_calibration.py for Food Lion.
# Only change: loads model_foodlion instead of model_approach2.
#   - Target  : 27 Food Lion stores (ground truth)
#   - Comps   : Harris Teeter + Publix + Aldi + Lidl + others
#
# OUTPUT:  Results_final/FL_results/V2/
#   calibration_report.txt
#   learned_weights.json
#   sensitivity_summary.csv
#   sensitivity_heatmap.csv
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations_foodlion/V2/V2_weight_calibration_fl.py
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

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import Food Lion model instead of HT model
from model_foodlion import (
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

RESULTS_DIR = "Results_final/FL_results/V2"
os.makedirs(RESULTS_DIR, exist_ok=True)

SPEED_MPS = 35 * MILE_M / 3600.0


# =============================================================
# FastScorer — identical to HT version, uses FL state
# =============================================================

class FastScorer:
    def __init__(self, state, radius_miles=5.0, K=3):
        print("[FastScorer] Pre-computing score components (Food Lion)...")
        t0 = time.perf_counter()

        self.radius_miles = radius_miles
        rad_m  = radius_miles * MILE_M
        cutoff = rad_m * CIRCUITY_FACTOR

        fl_m        = state["ht_m"]    # named ht_m internally, holds FL stores
        comp_m      = state["comp_m"]
        cands_m     = state["cands_m"]
        cands_ll    = state["cands_ll"].copy().reset_index(drop=True)
        bg_m        = state["bg_m"]
        road_csr       = state.get("road_csr",        None)
        road_kdtree    = state.get("road_kdtree",     None)
        bg_node_ids    = state.get("bg_node_ids",     None)
        bg_dist_matrix = state.get("bg_dist_matrix",  None)

        bg_cent = bg_m.copy()
        if "cent" not in bg_cent.columns:
            bg_cent["cent"] = bg_cent.geometry.centroid
        bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

        pop_col = _first_existing(bg_cent.columns,
                                  ["population","pop_total","pop","POP","B01001_001E"])
        if pop_col is None:
            raise KeyError("No population column in bg_m.")
        bg_pop = pd.to_numeric(bg_cent[pop_col], errors="coerce").fillna(0).to_numpy()

        inc_col = _first_existing(bg_cent.columns,
                                  ["income","median_income","med_income","B19013_001E"])
        bg_inc  = (pd.to_numeric(bg_cent[inc_col], errors="coerce").fillna(0).to_numpy()
                   if inc_col else np.zeros_like(bg_pop))

        dens_col = _first_existing(bg_cent.columns, ["pop_per_sqmi","density","DENSITY"])
        dens_w   = (norm_weight(pd.to_numeric(bg_cent[dens_col],
                                errors="coerce").fillna(0).to_numpy())
                    if dens_col else np.ones_like(bg_pop))

        block_weight = (0.4 * norm_weight(bg_pop)
                      + 0.3 * norm_weight(bg_inc)
                      + 0.3 * dens_w)

        bg_tree = cKDTree(bg_xy)

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

        xs = np.concatenate([fl_m.geometry.x.values, comp_m.geometry.x.values])
        ys = np.concatenate([fl_m.geometry.y.values, comp_m.geometry.y.values])
        all_store_tree = cKDTree(np.c_[xs, ys]) if len(xs) > 0 else None

        cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
        N_cands = len(cand_xy)
        print(f"[FastScorer] Computing components for {N_cands} candidates...")

        pop_arr    = np.zeros(N_cands)
        s10k_arr   = np.zeros(N_cands)
        access_arr = np.zeros(N_cands)

        self._bg_xy        = bg_xy
        self._bg_tree      = bg_tree
        self._bg_pop       = bg_pop
        self._block_weight = block_weight
        self._bg_tcomp     = bg_tcomp
        self._cand_xy      = cand_xy
        self._all_store_tree = all_store_tree
        self._road_kdtree  = road_kdtree
        self._rad_m        = rad_m
        self._cands_ll     = cands_ll

        self._cand_bg_idxs  = []
        self._cand_bg_tnew  = []
        self._cand_node_ids = []

        for i, (cx, cy) in enumerate(cand_xy):
            cand_node = (_nearest_node_idx(road_kdtree, cx, cy)
                         if road_kdtree is not None else None)
            self._cand_node_ids.append(cand_node)

            idxs = bg_tree.query_ball_point([cx, cy], r=rad_m)
            if idxs:
                idxs_arr = np.asarray(idxs, dtype=np.int32)
                pop_buf  = float(np.sum(bg_pop[idxs_arr]))
                pop_arr[i] = pop_buf

                if bg_dist_matrix is not None and cand_node is not None:
                    dist_m = bg_dist_matrix[idxs_arr, cand_node].astype(np.float64)
                    dist_m = np.where(np.isfinite(dist_m), dist_m,
                                      rad_m * CIRCUITY_FACTOR)
                    t_new  = (dist_m / SPEED_MPS) / 60.0
                else:
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

        print("[FastScorer] Computing accessibility...")
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

        print(f"[FastScorer] Accessibility done in {time.perf_counter()-t_acc:.2f}s")
        self._access_arr = access_arr
        print(f"[FastScorer] Pre-computation done in {time.perf_counter()-t0:.1f}s")

    def score(self, W1, W2, W3, beta):
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

        potential_norm = norm_weight(potential_arr)
        access_norm    = norm_weight(self._access_arr)
        s10k_norm      = norm_weight(self._s10k_arr)
        scores = W1 * potential_norm + W2 * access_norm - W3 * s10k_norm
        return scores, potential_norm

    def rank_of_locations(self, query_xy_m, scores):
        cand_tree   = cKDTree(self._cand_xy)
        score_order = np.argsort(-scores)
        rank_lookup = np.empty_like(score_order)
        rank_lookup[score_order] = np.arange(1, len(scores) + 1)
        ranks = []
        for (qx, qy) in query_xy_m:
            d, nearest = cand_tree.query([qx, qy], k=1)
            ranks.append({
                "rank":       int(rank_lookup[nearest]),
                "score":      float(scores[nearest]),
                "dist_m":     float(d),
                "percentile": float((1.0 - rank_lookup[nearest] / len(scores)) * 100),
            })
        return ranks


# =============================================================
# WEIGHT LEARNING
# =============================================================

def learn_weights(state, radius_miles=5.0, K=3, seed=42):
    print("\n" + "=" * 60)
    print("V2: DATA-DRIVEN WEIGHT LEARNING — Food Lion")
    print("=" * 60)

    scorer  = FastScorer(state, radius_miles=radius_miles, K=K)
    fl_m    = state["ht_m"].reset_index(drop=True)
    fl_xy_m = np.c_[fl_m.geometry.x.values, fl_m.geometry.y.values]

    n_fl    = len(fl_m)
    n_cands = len(scorer._cand_xy)
    print(f"\nGround truth: {n_fl} real Food Lion stores")
    print(f"Candidate pool: {n_cands} locations")

    eval_count = [0]

    def objective(params):
        W1, W2, W3, beta = params
        if W1 <= 0 or W2 <= 0 or W3 <= 0 or beta <= 0:
            return 1e9
        wsum = W1 + W2 + W3
        W1n, W2n, W3n = W1/wsum, W2/wsum, W3/wsum
        scores, _ = scorer.score(W1n, W2n, W3n, beta)
        results   = scorer.rank_of_locations(fl_xy_m, scores)
        mean_rank = float(np.mean([r["rank"] for r in results]))
        eval_count[0] += 1
        if eval_count[0] % 50 == 0:
            print(f"  eval {eval_count[0]:4d}: "
                  f"W=({W1n:.2f},{W2n:.2f},{W3n:.2f}) "
                  f"beta={beta:.2f} -> mean_rank={mean_rank:.1f} / {n_cands}")
        return mean_rank

    bounds = [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (1.0, 3.0)]

    print("\n--- Stage 1: Differential Evolution ---")
    t0 = time.perf_counter()
    de_result = differential_evolution(
        objective, bounds=bounds, seed=seed,
        maxiter=200, popsize=12, tol=0.01,
        mutation=(0.5, 1.5), recombination=0.7,
        workers=1, polish=False,
    )
    print(f"DE done in {time.perf_counter()-t0:.1f}s  best={de_result.fun:.2f}")

    print("\n--- Stage 2: Nelder-Mead refinement ---")
    nm_result = minimize(
        objective, x0=de_result.x, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 0.1},
    )
    print(f"NM done  best={nm_result.fun:.2f}  success={nm_result.success}")

    W1r, W2r, W3r, beta_r = nm_result.x
    wsum = W1r + W2r + W3r
    W1f, W2f, W3f = W1r/wsum, W2r/wsum, W3r/wsum
    beta_f = float(np.clip(beta_r, 1.0, 3.0))

    scores_learned, _ = scorer.score(W1f, W2f, W3f, beta_f)
    results_learned   = scorer.rank_of_locations(fl_xy_m, scores_learned)
    ranks_learned = [r["rank"]       for r in results_learned]
    pcts_learned  = [r["percentile"] for r in results_learned]

    scores_equal, _ = scorer.score(1/3, 1/3, 1/3, 2.0)
    results_equal   = scorer.rank_of_locations(fl_xy_m, scores_equal)
    ranks_equal     = [r["rank"] for r in results_equal]

    scores_paper, _ = scorer.score(0.4, 0.3, 0.3, 2.0)
    results_paper   = scorer.rank_of_locations(fl_xy_m, scores_paper)
    ranks_paper     = [r["rank"] for r in results_paper]

    report = {
        "brand": "Food Lion",
        "learned_weights": {
            "W1":   round(W1f,   4),
            "W2":   round(W2f,   4),
            "W3":   round(W3f,   4),
            "beta": round(beta_f, 4),
        },
        "optimisation": {
            "objective_value": round(float(nm_result.fun), 2),
            "n_evaluations":   int(eval_count[0]),
            "de_success":      bool(de_result.success),
            "nm_success":      bool(nm_result.success),
        },
        "fl_rank_stats": {
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
        "per_fl_store":  results_learned,
        "n_fl_stores":   n_fl,
        "n_candidates":  n_cands,
        "radius_miles":  radius_miles,
    }

    return report, scorer, scores_learned


# =============================================================
# SENSITIVITY ANALYSIS
# =============================================================

def sensitivity_analysis(scorer, learned_weights, n_top=10, min_sep_miles=2.5):
    print("\n" + "=" * 60)
    print("V2: SENSITIVITY ANALYSIS — Food Lion")
    print("=" * 60)

    lw = learned_weights
    W1_vals   = np.clip([lw["W1"] * f for f in [0.5, 0.7, 1.0, 1.3, 1.5]], 0.05, 0.90)
    W2_vals   = np.clip([lw["W2"] * f for f in [0.5, 0.7, 1.0, 1.3, 1.5]], 0.05, 0.90)
    beta_vals = [max(1.0, lw["beta"] + d) for d in [-0.5, -0.25, 0.0, 0.25, 0.5]]

    configs = []
    for W1, W2, beta in itertools.product(W1_vals, W2_vals, beta_vals):
        W3   = max(0.05, 1.0 - W1 - W2)
        wsum = W1 + W2 + W3
        configs.append((W1/wsum, W2/wsum, W3/wsum, beta))

    print(f"\nWeight configurations: {len(configs)}")
    min_sep_m = min_sep_miles * MILE_M
    cand_xy   = scorer._cand_xy
    n_cands   = len(cand_xy)
    appearance_count = np.zeros(n_cands, dtype=np.int32)
    grid_results     = []

    for i, (W1, W2, W3, beta) in enumerate(configs):
        scores, _ = scorer.score(W1, W2, W3, beta)
        order    = np.argsort(-scores)
        selected = []
        for idx in order:
            if len(selected) >= n_top:
                break
            if not selected:
                selected.append(idx); continue
            dx = cand_xy[idx, 0] - cand_xy[selected, 0]
            dy = cand_xy[idx, 1] - cand_xy[selected, 1]
            if np.all(np.hypot(dx, dy) >= min_sep_m):
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

    stability = appearance_count / len(configs)
    top_stable_idx = np.argsort(-stability)[:20]
    cand_ll = scorer._cands_ll
    stable_sites = []
    for idx in top_stable_idx:
        row  = cand_ll.iloc[int(idx)]
        geom = row.geometry
        stable_sites.append({
            "candidate_idx":      int(idx),
            "lat":                float(geom.y),
            "lon":                float(geom.x),
            "appearance_frac":    round(float(stability[idx]), 4),
            "appearance_pct":     round(float(stability[idx]) * 100, 1),
            "is_robust":          bool(stability[idx] >= 0.70),
            "n_configs_appeared": int(appearance_count[idx]),
            "n_configs_total":    len(configs),
        })

    n_robust = int(np.sum(stability >= 0.70))
    print(f"\nRobust sites (>=70%): {n_robust}")

    return {
        "n_configs":        len(configs),
        "n_robust_sites":   n_robust,
        "robust_threshold": 0.70,
        "top_stable_sites": stable_sites,
        "stability_array":  stability.tolist(),
        "grid_results":     grid_results,
    }


# =============================================================
# REPORT WRITER
# =============================================================

def write_report(opt1, opt2, output_dir=RESULTS_DIR):
    lw = opt1["learned_weights"]
    rs = opt1["fl_rank_stats"]
    ld = rs["learned"]

    lines = [
        "=" * 65,
        "  WEIGHT CALIBRATION REPORT — Food Lion",
        "  Food Lion Site Selection -- Charlotte, NC",
        "=" * 65,
        "\n--- DATA-DRIVEN WEIGHT LEARNING ---\n",
        f"  Brand calibrated against: Food Lion ({opt1['n_fl_stores']} stores)",
        f"  Candidate pool: {opt1['n_candidates']} locations",
        f"\n  Learned weights:",
        f"    W1 (market potential) = {lw['W1']:.4f}",
        f"    W2 (accessibility)    = {lw['W2']:.4f}",
        f"    W3 (competition)      = {lw['W3']:.4f}",
        f"    beta (Huff decay)     = {lw['beta']:.4f}",
        f"\n  Real FL store rank statistics:",
        f"  {'Config':<28} {'Mean Rank':>10} {'Median Rank':>12}",
        f"  {'-'*52}",
    ]
    for cfg, label in [("learned",       "Learned weights"),
                       ("paper_defaults", "Paper defaults (0.4/0.3/0.3)"),
                       ("equal_weights",  "Equal weights (1/3 each)")]:
        lines.append(f"  {label:<28} "
                     f"{rs[cfg]['mean_rank']:>10.1f} "
                     f"{rs[cfg]['median_rank']:>12.1f}")

    lines += [
        f"\n  With learned weights:",
        f"    Mean percentile of real FL stores: {ld['mean_percentile']:.1f}th",
        f"    Fraction in top 10%: {ld['pct_top10pct']:.0f}%",
        f"    Fraction in top 25%: {ld['pct_top25pct']:.0f}%",
        f"\n--- SENSITIVITY ANALYSIS ---",
        f"  Configurations tested: {opt2['n_configs']}",
        f"  Robust sites (>=70% appearance): {opt2['n_robust_sites']}",
    ]

    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "calibration_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Report] Saved to {path}")
    print(report_text)
    return report_text


# =============================================================
# MAIN
# =============================================================

def main():
    print("Loading Food Lion model state...")
    state = load_all()
    print(f"FL stores: {len(state['ht_m'])}, "
          f"Competitors: {len(state['comp_m'])}, "
          f"Candidates: {len(state['cands_m'])}")

    opt1, scorer, scores_learned = learn_weights(state, radius_miles=5.0, K=3, seed=42)

    lw_path = os.path.join(RESULTS_DIR, "learned_weights.json")
    with open(lw_path, "w") as f:
        json.dump(opt1["learned_weights"], f, indent=2)
    print(f"\n[Weights] Saved to {lw_path}")

    opt2 = sensitivity_analysis(scorer, opt1["learned_weights"])

    stable_df = pd.DataFrame(opt2["top_stable_sites"])
    stable_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_summary.csv"), index=False)

    grid_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "top_candidates"}
        for r in opt2["grid_results"]
    ])
    grid_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_heatmap.csv"), index=False)

    write_report(opt1, opt2)

    # Also save full JSON report
    full_report = {**opt1, "sensitivity": opt2}
    with open(os.path.join(RESULTS_DIR, "calibration_full.json"), "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n[Done] All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

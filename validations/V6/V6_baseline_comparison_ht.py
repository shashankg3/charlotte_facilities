# =============================================================
# V6_baseline_comparison_ht.py  (Harris Teeter study)
#
# Baseline comparison: does our model beat the simplest possible
# alternative?
#
# THREE BASELINES tested:
#   Baseline 1 — Population density only
#   Baseline 2 — Random
#   Baseline 3 — Inverse competitor count only
#
#   Our model — Huff + Dijkstra + Competition penalty
#   Reads optimal params from Results_final/V5_results/optimal_params.json
#   Falls back to V2 calibrated weights if V5 not yet run.
#
# OUTPUT:
#   Results_final/V6_results/baseline_comparison_results.json
#   Results_final/V6_results/baseline_comparison_report.txt
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations/V6/V6_baseline_comparison_ht.py
# =============================================================

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_approach2 import load_all, score_all_candidates_like_ht, MILE_M

RESULTS_DIR    = "Results_final/V6_results"
V5_PARAMS_FILE = "Results_final/V5_results/optimal_params.json"
V2_PARAMS_FILE = "Results_final/V2_results/learned_weights.json"
MAX_MATCH_MI   = 0.5
N_PERMUTATIONS = 10_000
RANDOM_SEED    = 42


# =============================================================
# HELPERS
# =============================================================

def percentile_ranks(scores):
    valid = np.isfinite(scores)
    out   = np.full(len(scores), np.nan)
    vs    = scores[valid]
    if len(vs) < 2:
        return out
    ranks = (np.argsort(np.argsort(vs)).astype(float) / (len(vs) - 1)) * 100.0
    out[valid] = ranks
    return out


def fl_mean_pct(pct_full, fl_idxs, close):
    matched  = fl_idxs[close]
    fl_pcts  = pct_full[matched]
    fl_valid = fl_pcts[np.isfinite(fl_pcts)]
    return float(np.mean(fl_valid)) if len(fl_valid) >= 3 else np.nan


def permutation_test(pct_full, fl_idxs, close, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    valid_idx  = np.where(np.isfinite(pct_full))[0]
    observed   = fl_mean_pct(pct_full, fl_idxs, close)
    n_fl       = int(close.sum())
    rng        = np.random.default_rng(seed)
    perm_means = np.array([
        np.mean(pct_full[rng.choice(valid_idx, size=n_fl, replace=False)])
        for _ in range(n_perm)
    ])
    p_value = float(np.sum(perm_means >= observed) / n_perm)
    return p_value, float(np.mean(perm_means)), float(np.std(perm_means))


def run_baseline(name, raw_scores, fl_idxs, close):
    pct   = percentile_ranks(raw_scores)
    mean  = fl_mean_pct(pct, fl_idxs, close)
    pval, pm, ps = permutation_test(pct, fl_idxs, close)
    sig   = pval < 0.05
    print(f"  {name:<35} mean_pct={mean:5.1f}th  "
          f"p={pval:.4f}  {'SIGNIFICANT' if sig else 'not significant'}")
    return {
        "name":             name,
        "mean_pct_rank":    round(mean, 2),
        "permutation_mean": round(pm, 2),
        "permutation_std":  round(ps, 2),
        "p_value":          round(pval, 4),
        "p_value_str":      "< 0.001" if pval < 0.001 else f"{pval:.4f}",
        "significant":      sig,
    }


# =============================================================
# MAIN
# =============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 65)
    print("  V6: BASELINE COMPARISON — Harris Teeter")
    print("=" * 65)

    # ── Load params (V5 preferred, V2 fallback) ───────────────
    if os.path.exists(V5_PARAMS_FILE):
        with open(V5_PARAMS_FILE) as f:
            p = json.load(f)
        print(f"\n  Params source: V5 LOOCV-validated")
    elif os.path.exists(V2_PARAMS_FILE):
        with open(V2_PARAMS_FILE) as f:
            lw = json.load(f)
        p = {
            "radius_miles": 5.0,
            "beta":         float(lw.get("beta", 2.0)),
            "K":            3,
            "W1":           float(lw["W1"]),
            "W2":           float(lw["W2"]),
            "W3":           float(lw["W3"]),
        }
        print(f"\n  Params source: V2 calibrated weights (run V5 search 3 for optimal)")
    else:
        p = {"radius_miles": 5.0, "beta": 2.0, "K": 3,
             "W1": 0.4, "W2": 0.3, "W3": 0.3}
        print(f"\n  Params source: defaults (run V2 or V5 first)")

    print(f"  r={p['radius_miles']}mi  beta={p['beta']}  K={p['K']}")
    print(f"  W1={p['W1']}  W2={p['W2']}  W3={p['W3']}")
    radius_m = p["radius_miles"] * MILE_M

    # ── Load model state ──────────────────────────────────────
    print("\n[1/3] Loading model state...")
    state   = load_all()
    ht_m    = state["ht_m"]
    cands_m = state["cands_m"]
    bg_m    = state["bg_m"]

    cand_xy   = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    ht_xy     = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    bg_xy     = np.c_[bg_m.geometry.centroid.x.values,
                       bg_m.geometry.centroid.y.values]

    cand_tree         = cKDTree(cand_xy)
    ht_dists, ht_idxs = cand_tree.query(ht_xy)
    close             = ht_dists <= (MAX_MATCH_MI * MILE_M)
    n_matched         = int(close.sum())
    print(f"  HT stores matched (within 0.5mi): {n_matched}/{len(ht_m)}")

    pop_col = None
    for col in ["population", "pop", "total_pop", "POP"]:
        if col in bg_m.columns:
            pop_col = col
            break
    bg_pop = bg_m[pop_col].fillna(0).values.astype(float) \
             if pop_col else np.ones(len(bg_m))

    comp_xy   = np.c_[state["comp_m"].geometry.x.values,
                       state["comp_m"].geometry.y.values]
    comp_tree = cKDTree(comp_xy)
    bg_tree   = cKDTree(bg_xy)

    # ── Compute baseline scores ───────────────────────────────
    print("\n[2/3] Computing baseline scores for all candidates...")
    pop_scores  = np.zeros(len(cands_m))
    comp_scores = np.zeros(len(cands_m))

    for i, (cx, cy) in enumerate(cand_xy):
        bg_idxs = bg_tree.query_ball_point([cx, cy], r=radius_m)
        if bg_idxs:
            pop_scores[i] = float(np.sum(bg_pop[bg_idxs]))
        n_comp = len(comp_tree.query_ball_point([cx, cy], r=radius_m))
        comp_scores[i] = 1.0 / (n_comp + 1)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(cands_m)} candidates processed...")

    print("  Done.")

    # ── Run full model ────────────────────────────────────────
    print("\n[3/3] Running full model + all baselines...")
    _, _, _, out_ll = score_all_candidates_like_ht(
        state,
        radius_miles=p["radius_miles"], beta=p["beta"], K=p["K"],
        W1=p["W1"], W2=p["W2"], W3=p["W3"],
        return_all=True,
    )
    model_scores = out_ll["pair_score"].values
    rng          = np.random.default_rng(RANDOM_SEED)
    rand_scores  = rng.random(len(cands_m))

    # ── Evaluate all four ─────────────────────────────────────
    print(f"\n  {'Method':<35} {'MeanPct':>8}  {'p-value':>8}  Significant?")
    print(f"  {'-'*65}")

    results = []
    results.append(run_baseline("Random (expected ~50th)",
                                rand_scores,  ht_idxs, close))
    results.append(run_baseline("Population density only",
                                pop_scores,   ht_idxs, close))
    results.append(run_baseline("Inverse competitor count only",
                                comp_scores,  ht_idxs, close))
    results.append(run_baseline("Our model (Huff+Dijkstra+Comp)",
                                model_scores, ht_idxs, close))

    our   = results[3]
    pop   = results[1]
    comp_ = results[2]
    rand  = results[0]

    print(f"\n{'=' * 65}")
    print(f"  COMPARISON SUMMARY — Harris Teeter")
    print(f"{'=' * 65}")
    print(f"  Random baseline          : {rand['mean_pct_rank']:5.1f}th pctile  "
          f"p={rand['p_value_str']}")
    print(f"  Population density only  : {pop['mean_pct_rank']:5.1f}th pctile  "
          f"p={pop['p_value_str']}")
    print(f"  Competitor count only    : {comp_['mean_pct_rank']:5.1f}th pctile  "
          f"p={comp_['p_value_str']}")
    print(f"  Our model                : {our['mean_pct_rank']:5.1f}th pctile  "
          f"p={our['p_value_str']}  <- best")
    print(f"")
    print(f"  Gain over population     : +{our['mean_pct_rank']-pop['mean_pct_rank']:.1f} pctile points")
    print(f"  Gain over comp-only      : +{our['mean_pct_rank']-comp_['mean_pct_rank']:.1f} pctile points")
    print(f"  Gain over random         : +{our['mean_pct_rank']-rand['mean_pct_rank']:.1f} pctile points")
    print(f"{'=' * 65}")
    print(f"  Total runtime: {time.perf_counter()-t0:.1f}s")

    output = {
        "brand":           "Harris Teeter",
        "city":            "Charlotte, NC",
        "test":            "baseline_comparison_v6_ht",
        "model_params":    p,
        "n_ht_matched":    n_matched,
        "n_permutations":  N_PERMUTATIONS,
        "results":         results,
        "gain_over_population": round(our["mean_pct_rank"] - pop["mean_pct_rank"],  2),
        "gain_over_comp_only":  round(our["mean_pct_rank"] - comp_["mean_pct_rank"], 2),
        "gain_over_random":     round(our["mean_pct_rank"] - rand["mean_pct_rank"],  2),
    }

    json_path = os.path.join(RESULTS_DIR, "baseline_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    report = [
        "V6 BASELINE COMPARISON REPORT -- Harris Teeter Charlotte",
        "=" * 65,
        "",
        "QUESTION: Does the model beat simple alternatives?",
        "",
        f"  {'Method':<35} {'MeanPct':>8}  {'p-value':>8}",
        "  " + "-" * 55,
    ]
    for r in results:
        report.append(
            f"  {r['name']:<35} {r['mean_pct_rank']:>8.2f}  {r['p_value_str']:>8}"
        )
    report += [
        "",
        "GAINS OVER BASELINES",
        f"  vs population density : +{output['gain_over_population']:.1f} pctile points",
        f"  vs competitor count   : +{output['gain_over_comp_only']:.1f} pctile points",
        f"  vs random             : +{output['gain_over_random']:.1f} pctile points",
    ]

    txt_path = os.path.join(RESULTS_DIR, "baseline_comparison_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()

# =============================================================
# V5_hyperparam_search_3_ht.py  (Harris Teeter study)
#
# SEARCH 3 — Fine grid + Leave-One-Out Cross Validation (LOOCV)
#
# WHY LOOCV:
#   Searches 1 & 2 used mean-percentile-rank as the objective.
#   This can overfit — weights that rank ALL real HT stores high
#   simultaneously may just be tuned to these specific stores.
#   LOOCV hides one store at a time and checks if the model
#   still ranks it highly. A model that generalises will score
#   well on LOOCV; one that overfits will not.
#
# PIPELINE:
#   Stage A — Fine coarse grid around FL winner as starting point
#             radius {4.5,5.0,5.5,6.0,6.5,7.0} x beta {2.5,3.0,3.5,4.0}
#             x K {4,5,6,7}  with W=0.4/0.3/0.3 (HT balanced default)
#             Objective: mean percentile rank
#             → top-20 candidates forwarded to LOOCV
#
#   Stage B — LOOCV on top-20 Huff param combos
#             For each combo: run model once per HT store (hide one
#             store each time), record rank of hidden store
#             Objective: mean LOOCV percentile
#             → best (radius, beta, K) confirmed
#
#   Stage C — Weight search at LOOCV-confirmed Huff params
#             Fine W grid, mean-percentile objective
#
#   Stage D — LOOCV on top-10 weight combos to confirm
#
# OUTPUT:
#   Results_final/V5_results/hyperparam_search_3_results.json
#   Results_final/V5_results/hyperparam_search_3_report.txt
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations/V5/V5_hyperparam_search_3_ht.py
#
# ESTIMATED RUNTIME: 20-35 minutes
# =============================================================

import os, sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_approach2 import load_all, score_all_candidates_like_ht, MILE_M

RESULTS_DIR  = "Results_final/V5_results"
MAX_MATCH_MI = 0.5
LOOCV_TOP_N  = 20      # run LOOCV only on top-N from coarse grid

# ── Fine grids (sweeping 4–7mi, centred on search_2 winner r=6.0) ─
RADIUS_GRID = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
BETA_GRID   = [2.5, 3.0, 3.5]
K_GRID      = [3, 4, 5, 6]

W1_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
W2_GRID = [0.1, 0.2, 0.3, 0.4]


# =============================================================
# HELPERS
# =============================================================

def _score_candidates(state, radius, beta, K, W1, W2, W3):
    """Run model, return (scores array, valid_mask). None on error."""
    W3 = round(1.0 - W1 - W2, 6)
    if W3 < 0.04:
        return None, None
    try:
        _, _, _, out_ll = score_all_candidates_like_ht(
            state, radius_miles=radius, beta=beta, K=K,
            W1=W1, W2=W2, W3=W3, return_all=True,
        )
    except Exception:
        return None, None
    scores = out_ll["pair_score"].values
    valid  = np.isfinite(scores)
    return scores, valid


def mean_pct_rank(state, radius, beta, K, W1, W2, W3, ht_xy, cand_tree):
    """Mean percentile rank of matched HT stores. Higher = better."""
    scores, valid = _score_candidates(state, radius, beta, K, W1, W2, W3)
    if scores is None or valid.sum() < 10:
        return np.nan

    vs      = scores[valid]
    n       = len(vs)
    ranks   = (np.argsort(np.argsort(vs)).astype(float) / (n-1)) * 100.0
    pct_full = np.full(len(scores), np.nan)
    pct_full[valid] = ranks

    ht_dists, ht_idxs = cand_tree.query(ht_xy)
    close   = ht_dists <= (MAX_MATCH_MI * MILE_M)
    matched = ht_idxs[close]
    ht_pcts = pct_full[matched]
    good    = ht_pcts[np.isfinite(ht_pcts)]
    return float(np.mean(good)) if len(good) >= 5 else np.nan


def loocv_score(state, radius, beta, K, W1, W2, W3, ht_m, cand_tree):
    """
    Leave-One-Out CV score.
    For each HT store: temporarily remove it from state["ht_m"],
    run model, find where that store ranks among all candidates.
    Returns mean percentile rank across all left-out stores.
    """
    W3 = round(1.0 - W1 - W2, 6)
    if W3 < 0.04:
        return np.nan

    import geopandas as gpd
    ht_xy_all = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    cand_xy   = np.c_[state["cands_m"].geometry.x.values,
                       state["cands_m"].geometry.y.values]

    # Pre-match HT stores → nearest candidates (done once)
    ctree = cKDTree(cand_xy)
    ht_dists, ht_idxs = ctree.query(ht_xy_all)
    valid_ht = np.where(ht_dists <= (MAX_MATCH_MI * MILE_M))[0]

    if len(valid_ht) < 5:
        return np.nan

    held_out_pcts = []
    original_ht   = state["ht_m"]

    for leave_idx in valid_ht:
        # Remove this one HT store from the target set
        mask         = np.ones(len(original_ht), dtype=bool)
        mask[leave_idx] = False
        state["ht_m"] = original_ht.iloc[mask].copy()

        scores, valid = _score_candidates(state, radius, beta, K, W1, W2, W3)
        state["ht_m"] = original_ht   # restore immediately

        if scores is None or valid.sum() < 10:
            continue

        vs      = scores[valid]
        n       = len(vs)
        ranks   = (np.argsort(np.argsort(vs)).astype(float) / (n-1)) * 100.0
        pct_full = np.full(len(scores), np.nan)
        pct_full[valid] = ranks

        cand_idx = ht_idxs[leave_idx]
        pct      = pct_full[cand_idx]
        if np.isfinite(pct):
            held_out_pcts.append(float(pct))

    if len(held_out_pcts) < 5:
        return np.nan
    return float(np.mean(held_out_pcts))


# =============================================================
# MAIN
# =============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    T_START = time.perf_counter()

    print("=" * 65)
    print("  V5 SEARCH 3: Fine Grid + LOOCV — Harris Teeter")
    print("=" * 65)
    print("  Objective: Leave-One-Out CV (generalisation, not overfitting)")

    # ── Load model ────────────────────────────────────────────
    print("\n[LOAD] Loading model state...")
    state   = load_all()
    ht_m    = state["ht_m"]
    cands   = state["cands_m"]
    cand_xy = np.c_[cands.geometry.x.values, cands.geometry.y.values]
    cand_tree = cKDTree(cand_xy)
    ht_xy   = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    print(f"       Candidates: {len(cands)}, HT stores: {len(ht_m)}")
    print(f"       [loaded in {time.perf_counter()-T_START:.1f}s]")

    # ── Stage A: Fine coarse grid (mean-pct objective) ────────
    W1_FIXED, W2_FIXED, W3_FIXED = 0.6, 0.3, 0.1   # search_2 winner
    combos_a = list(itertools.product(RADIUS_GRID, BETA_GRID, K_GRID))
    print(f"\n[STAGE A] Fine Huff grid — {len(combos_a)} combos "
          f"(W={W1_FIXED}/{W2_FIXED}/{W3_FIXED})")

    stage_a = []
    t = time.perf_counter()
    for i, (r, b, k) in enumerate(combos_a, 1):
        mpr = mean_pct_rank(state, r, b, k,
                            W1_FIXED, W2_FIXED, W3_FIXED, ht_xy, cand_tree)
        stage_a.append({"radius": r, "beta": b, "K": k,
                        "mean_pct": round(mpr, 3) if np.isfinite(mpr) else None})
        if i % 15 == 0 or i == len(combos_a):
            best_so_far = max((x for x in stage_a if x["mean_pct"]),
                              key=lambda x: x["mean_pct"])
            print(f"  {i:3d}/{len(combos_a)}  "
                  f"best: r={best_so_far['radius']} b={best_so_far['beta']} "
                  f"K={best_so_far['K']} → {best_so_far['mean_pct']}th  "
                  f"[{time.perf_counter()-t:.0f}s]")

    valid_a  = [x for x in stage_a if x["mean_pct"] is not None]
    top20_a  = sorted(valid_a, key=lambda x: x["mean_pct"], reverse=True)[:LOOCV_TOP_N]
    print(f"\n  Top-{LOOCV_TOP_N} forwarded to LOOCV:")
    for x in top20_a[:5]:
        print(f"    r={x['radius']} b={x['beta']} K={x['K']} "
              f"→ {x['mean_pct']}th pctile")
    print(f"    ...")

    # ── Stage B: LOOCV on top-20 Huff combos ─────────────────
    print(f"\n[STAGE B] LOOCV on top-{LOOCV_TOP_N} Huff combos "
          f"(~{LOOCV_TOP_N * len(ht_m)} model runs — slow)")

    stage_b = []
    t = time.perf_counter()
    best_loocv_score  = -np.inf
    best_loocv_params = None

    for i, combo in enumerate(top20_a, 1):
        r, b, k = combo["radius"], combo["beta"], combo["K"]
        lscore  = loocv_score(state, r, b, k,
                              W1_FIXED, W2_FIXED, W3_FIXED, ht_m, cand_tree)
        combo["loocv_pct"] = round(lscore, 3) if np.isfinite(lscore) else None
        stage_b.append(combo)

        if np.isfinite(lscore) and lscore > best_loocv_score:
            best_loocv_score  = lscore
            best_loocv_params = (r, b, k)

        elapsed = time.perf_counter() - t
        eta     = (elapsed / i) * (LOOCV_TOP_N - i)
        print(f"  {i:2d}/{LOOCV_TOP_N}  r={r} b={b} K={k}  "
              f"LOOCV={lscore:.2f}th  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    best_r, best_b, best_k = best_loocv_params
    print(f"\n  LOOCV winner: radius={best_r} beta={best_b} K={best_k} "
          f"→ {best_loocv_score:.3f}th pctile")

    # ── Stage C: Fine weight search ───────────────────────────
    w_combos = [(w1, w2, round(1-w1-w2, 4))
                for w1 in W1_GRID for w2 in W2_GRID
                if 0.04 <= round(1-w1-w2, 4) <= 0.85]
    print(f"\n[STAGE C] Weight search — {len(w_combos)} combos "
          f"(r={best_r} b={best_b} K={best_k})")

    stage_c = []
    t = time.perf_counter()
    best_w_score = -np.inf
    best_w       = None

    for i, (w1, w2, w3) in enumerate(w_combos, 1):
        mpr = mean_pct_rank(state, best_r, best_b, best_k,
                            w1, w2, w3, ht_xy, cand_tree)
        stage_c.append({"radius": best_r, "beta": best_b, "K": best_k,
                        "W1": w1, "W2": w2, "W3": w3,
                        "mean_pct": round(mpr, 3) if np.isfinite(mpr) else None})
        if np.isfinite(mpr) and mpr > best_w_score:
            best_w_score = mpr
            best_w       = (w1, w2, w3)
        if i % 10 == 0 or i == len(w_combos):
            print(f"  {i:3d}/{len(w_combos)}  "
                  f"best: W1={best_w[0]} W2={best_w[1]} W3={best_w[2]} "
                  f"→ {best_w_score:.3f}th  [{time.perf_counter()-t:.0f}s]")

    valid_c  = [x for x in stage_c if x["mean_pct"] is not None]
    top10_c  = sorted(valid_c, key=lambda x: x["mean_pct"], reverse=True)[:10]

    # ── Stage D: LOOCV on top-10 weight combos ───────────────
    print(f"\n[STAGE D] LOOCV on top-10 weight combos")

    stage_d = []
    t = time.perf_counter()
    best_final_score  = -np.inf
    best_final_params = None

    for i, combo in enumerate(top10_c, 1):
        w1, w2, w3 = combo["W1"], combo["W2"], combo["W3"]
        lscore = loocv_score(state, best_r, best_b, best_k,
                             w1, w2, w3, ht_m, cand_tree)
        combo["loocv_pct"] = round(lscore, 3) if np.isfinite(lscore) else None
        stage_d.append(combo)

        if np.isfinite(lscore) and lscore > best_final_score:
            best_final_score  = lscore
            best_final_params = (w1, w2, w3)

        elapsed = time.perf_counter() - t
        eta     = (elapsed / i) * (10 - i)
        print(f"  {i:2d}/10  W1={w1} W2={w2} W3={w3}  "
              f"LOOCV={lscore:.2f}th  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    best_w1, best_w2, best_w3 = best_final_params

    # ── Summary ───────────────────────────────────────────────
    total_time = time.perf_counter() - T_START
    best = {
        "radius_miles":      best_r,
        "beta":              best_b,
        "K":                 best_k,
        "W1":                best_w1,
        "W2":                best_w2,
        "W3":                best_w3,
        "mean_pct_rank":     round(best_w_score, 3),
        "loocv_pct_rank":    round(best_final_score, 3),
    }

    print(f"\n{'=' * 65}")
    print(f"  FINAL BEST COMBINATION  (LOOCV-validated)")
    print(f"{'=' * 65}")
    print(f"  Catchment radius : {best_r} miles")
    print(f"  Beta (decay)     : {best_b}")
    print(f"  K (competitors)  : {best_k}")
    print(f"  W1 (Huff demand) : {best_w1}")
    print(f"  W2 (Dijkstra)    : {best_w2}")
    print(f"  W3 (Comp penalty): {best_w3}")
    print(f"  Mean pct rank    : {best['mean_pct_rank']}th  (all stores)")
    print(f"  LOOCV pct rank   : {best['loocv_pct_rank']}th  (held-out stores)")
    print(f"  Total runtime    : {total_time/60:.1f} min")
    print(f"{'=' * 65}")

    print(f"\n  Progress vs earlier searches:")
    print(f"    search_1 (mean-pct)  : 60.63th pctile")
    print(f"    search_2 (coord.desc): 60.958th pctile")
    print(f"    search_3 (LOOCV val) : {best['loocv_pct_rank']}th  ← generalises to unseen stores")

    print(f"\n  Interpretation:")
    print(f"    LOOCV score {best['loocv_pct_rank']:.1f}th means: when we hide any one")
    print(f"    real HT store and run the model blind, it still ranks")
    print(f"    that store in the top {100-best['loocv_pct_rank']:.0f}% of all candidates on average.")
    print(f"    This is the generalisation-validated result.")

    # Top-10 Stage D
    top10_final = sorted(
        [x for x in stage_d if x["loocv_pct"] is not None],
        key=lambda x: x["loocv_pct"], reverse=True
    )[:10]

    print(f"\n  Top-10 weight combos (LOOCV-ranked):")
    print(f"  {'Rank':<5} {'W1':>5} {'W2':>5} {'W3':>5} "
          f"{'MeanPct':>9} {'LOOCV':>8}")
    print(f"  {'-'*45}")
    for i, r in enumerate(top10_final, 1):
        print(f"  {i:<5} {r['W1']:>5.2f} {r['W2']:>5.2f} {r['W3']:>5.2f} "
              f"{r['mean_pct']:>9.3f} {r['loocv_pct']:>8.3f}")

    # ── Save ─────────────────────────────────────────────────
    output = {
        "brand":        "Harris Teeter",
        "city":         "Charlotte, NC",
        "test":         "hyperparam_search_3_loocv",
        "best_params":  best,
        "stage_a_top20": top20_a,
        "stage_b_loocv": stage_b,
        "stage_c_top10_weights": top10_c,
        "stage_d_loocv_weights": top10_final,
        "runtime_minutes": round(total_time / 60, 1),
    }

    json_path = os.path.join(RESULTS_DIR, "hyperparam_search_3_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    report = [
        "V5 HYPERPARAM SEARCH 3 — Fine Grid + LOOCV — Harris Teeter Charlotte",
        "=" * 65,
        "",
        "METHOD: Fine coarse grid → LOOCV validation",
        "  Stage A: fine Huff grid (mean-pct), top-20 forwarded to LOOCV",
        "  Stage B: LOOCV on top-20 Huff combos → confirmed (r,b,K)",
        "  Stage C: fine weight grid at best (r,b,K)",
        "  Stage D: LOOCV on top-10 weight combos → final params",
        "",
        "LOOCV INTERPRETATION",
        "  For each real HT store, we hide it and check",
        "  if the model still ranks it highly. A high LOOCV score",
        "  means the model generalises — it's not just memorising",
        "  the existing stores.",
        "",
        f"FINAL RESULT",
        f"  radius={best_r}mi  beta={best_b}  K={best_k}",
        f"  W1={best_w1} (Huff demand)  "
        f"W2={best_w2} (Dijkstra)  W3={best_w3} (Comp penalty)",
        f"  Mean percentile rank : {best['mean_pct_rank']}th  (all stores)",
        f"  LOOCV percentile rank: {best['loocv_pct_rank']}th  (held-out validation)",
        "",
        "COMPARISON",
        "  search_1 (mean-pct only)       : 60.63th",
        "  search_2 (coordinate descent)  : 60.958th",
        f"  search_3 (LOOCV-validated)     : {best['loocv_pct_rank']}th",
        "",
        "TOP-10 WEIGHT COMBOS (LOOCV-ranked)",
        f"  {'Rank':<5} {'W1':>5} {'W2':>5} {'W3':>5} {'MeanPct':>9} {'LOOCV':>8}",
        "  " + "-" * 45,
    ]
    for i, r in enumerate(top10_final, 1):
        report.append(
            f"  {i:<5} {r['W1']:>5.2f} {r['W2']:>5.2f} {r['W3']:>5.2f} "
            f"{r['mean_pct']:>9.3f} {r['loocv_pct']:>8.3f}"
        )

    txt_path = os.path.join(RESULTS_DIR, "hyperparam_search_3_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # Write optimal_params.json for V4 and V6 to read
    optimal = {
        "source":          "V5_hyperparam_search_3_loocv",
        "radius_miles":    best_r,
        "beta":            best_b,
        "K":               best_k,
        "W1":              best_w1,
        "W2":              best_w2,
        "W3":              best_w3,
        "mean_pct_rank":   round(best_w_score, 3),
        "loocv_pct_rank":  round(best_final_score, 3),
    }
    opt_path = os.path.join(RESULTS_DIR, "optimal_params.json")
    with open(opt_path, "w") as f:
        json.dump(optimal, f, indent=2)

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    print(f"  Saved: {opt_path}  <- used by V4 and V6")
    print(f"\n  These are your final, generalisation-validated parameters.")


if __name__ == "__main__":
    main()

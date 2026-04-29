# =============================================================
# V5_hyperparam_search_4_ht.py  (Harris Teeter study)
#
# SEARCH 4 — Extended grid below radius=4.0 and above W1=0.70
#
# WHY SEARCH 4:
#   Search 3 winner hit BOTH grid edges:
#     radius = 4.0  (bottom of 4.0–7.0 grid)
#     W1     = 0.70 (top of 0.3–0.7 grid)
#   This means the true optimum likely lies outside the search_3
#   grid. Search 4 extends both edges and re-runs LOOCV to find
#   the true best.
#
# NEW GRIDS:
#   radius : 2.5, 3.0, 3.5, 4.0, 4.5  (extends below 4.0)
#   beta   : 3.0, 3.5, 4.0             (centred on search_3 winner 3.5)
#   K      : 5, 6, 7                    (centred on search_3 winner 6)
#   W1     : 0.65, 0.70, 0.75, 0.80, 0.85  (extends above 0.70)
#   W2     : 0.05, 0.10, 0.15, 0.20        (centred on search_3 winner 0.10)
#
# PIPELINE: identical to search_3
#   Stage A — coarse grid (mean-pct) → top-20
#   Stage B — LOOCV on top-20 Huff combos
#   Stage C — weight search at LOOCV winner
#   Stage D — LOOCV on top-10 weight combos
#
# CONVERGENCE CHECK:
#   If winner is no longer at a grid edge → converged, use these params
#   If still at edge → note direction for search_5
#
# OUTPUT:
#   Results_final/V5_results/hyperparam_search_4_results.json
#   Results_final/V5_results/hyperparam_search_4_report.txt
#   Results_final/V5_results/optimal_params.json  (updated if improved)
#
# USAGE:
#   cd <project_root>
#   C:\Python312\python.exe validations/V5/V5_hyperparam_search_4_ht.py
# =============================================================

import os, sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_approach2 import load_all, score_all_candidates_like_ht, MILE_M

RESULTS_DIR  = "Results_final/V5_results"
MAX_MATCH_MI = 0.5
LOOCV_TOP_N  = 20

# ── Extended grids ────────────────────────────────────────────
RADIUS_GRID = [2.5, 3.0, 3.5, 4.0, 4.5]   # extends below 4.0
BETA_GRID   = [3.0, 3.5, 4.0]              # centred on search_3 winner
K_GRID      = [5, 6, 7]                    # centred on search_3 winner

W1_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]  # extends above 0.70
W2_GRID = [0.05, 0.10, 0.15, 0.20]        # centred on search_3 winner

# Search_3 reference for comparison
SEARCH3_LOOCV = 64.706
SEARCH3_PARAMS = "r=4.0 b=3.5 K=6 W1=0.70 W2=0.10 W3=0.20"


# =============================================================
# HELPERS  (identical to search_3)
# =============================================================

def _score_candidates(state, radius, beta, K, W1, W2, W3):
    W3 = round(1.0 - W1 - W2, 6)
    if W3 < 0.02:
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
    fl_pcts = pct_full[matched]
    good    = fl_pcts[np.isfinite(fl_pcts)]
    return float(np.mean(good)) if len(good) >= 5 else np.nan


def loocv_score(state, radius, beta, K, W1, W2, W3, ht_m, cand_tree):
    W3 = round(1.0 - W1 - W2, 6)
    if W3 < 0.02:
        return np.nan

    ht_xy_all = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    cand_xy   = np.c_[state["cands_m"].geometry.x.values,
                       state["cands_m"].geometry.y.values]
    ctree = cKDTree(cand_xy)
    ht_dists, ht_idxs = ctree.query(ht_xy_all)
    valid_ht = np.where(ht_dists <= (MAX_MATCH_MI * MILE_M))[0]

    if len(valid_ht) < 5:
        return np.nan

    held_out_pcts = []
    original_ht   = state["ht_m"]

    for leave_idx in valid_ht:
        mask             = np.ones(len(original_ht), dtype=bool)
        mask[leave_idx]  = False
        state["ht_m"]    = original_ht.iloc[mask].copy()

        scores, valid = _score_candidates(state, radius, beta, K, W1, W2, W3)
        state["ht_m"] = original_ht

        if scores is None or valid.sum() < 10:
            continue

        vs       = scores[valid]
        n        = len(vs)
        ranks    = (np.argsort(np.argsort(vs)).astype(float) / (n-1)) * 100.0
        pct_full = np.full(len(scores), np.nan)
        pct_full[valid] = ranks

        pct = pct_full[ht_idxs[leave_idx]]
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
    print("  V5 SEARCH 4: Extended Grid + LOOCV — Harris Teeter")
    print("=" * 65)
    print(f"  Search 3 reference: LOOCV={SEARCH3_LOOCV}th  ({SEARCH3_PARAMS})")
    print(f"  Search 4 extends: radius down to 2.5mi, W1 up to 0.85")

    # ── Load model ────────────────────────────────────────────
    print("\n[LOAD] Loading model state...")
    state     = load_all()
    ht_m      = state["ht_m"]
    cands     = state["cands_m"]
    cand_xy   = np.c_[cands.geometry.x.values, cands.geometry.y.values]
    cand_tree = cKDTree(cand_xy)
    ht_xy     = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    print(f"       Candidates: {len(cands)}, HT stores: {len(ht_m)}")
    print(f"       [loaded in {time.perf_counter()-T_START:.1f}s]")

    # ── Stage A: coarse grid (mean-pct) ──────────────────────
    W1_FIXED, W2_FIXED, W3_FIXED = 0.70, 0.10, 0.20   # search_3 winner
    combos_a = list(itertools.product(RADIUS_GRID, BETA_GRID, K_GRID))
    print(f"\n[STAGE A] Extended Huff grid — {len(combos_a)} combos "
          f"(W={W1_FIXED}/{W2_FIXED}/{W3_FIXED})")

    stage_a = []
    t = time.perf_counter()
    for i, (r, b, k) in enumerate(combos_a, 1):
        mpr = mean_pct_rank(state, r, b, k,
                            W1_FIXED, W2_FIXED, W3_FIXED, ht_xy, cand_tree)
        stage_a.append({"radius": r, "beta": b, "K": k,
                        "mean_pct": round(mpr, 3) if np.isfinite(mpr) else None})
        if i % 15 == 0 or i == len(combos_a):
            valid_so_far = [x for x in stage_a if x["mean_pct"]]
            if valid_so_far:
                best_so_far = max(valid_so_far, key=lambda x: x["mean_pct"])
                print(f"  {i:3d}/{len(combos_a)}  "
                      f"best: r={best_so_far['radius']} b={best_so_far['beta']} "
                      f"K={best_so_far['K']} -> {best_so_far['mean_pct']}th  "
                      f"[{time.perf_counter()-t:.0f}s]")

    valid_a = [x for x in stage_a if x["mean_pct"] is not None]
    top20_a = sorted(valid_a, key=lambda x: x["mean_pct"], reverse=True)[:LOOCV_TOP_N]
    print(f"\n  Top-5 forwarded to LOOCV:")
    for x in top20_a[:5]:
        print(f"    r={x['radius']} b={x['beta']} K={x['K']} "
              f"-> {x['mean_pct']}th pctile")

    # ── Stage B: LOOCV on top-20 Huff combos ─────────────────
    print(f"\n[STAGE B] LOOCV on top-{LOOCV_TOP_N} Huff combos ...")
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
          f"-> {best_loocv_score:.3f}th pctile")

    # Convergence check
    at_radius_edge = (best_r == min(RADIUS_GRID) or best_r == max(RADIUS_GRID))
    if at_radius_edge:
        edge_dir = "below" if best_r == min(RADIUS_GRID) else "above"
        print(f"  WARNING: radius={best_r} is at grid edge ({edge_dir}) "
              f"-- consider extending grid further")
    else:
        print(f"  radius={best_r} is interior to grid -- good")

    # ── Stage C: Weight search ────────────────────────────────
    w_combos = [(w1, w2, round(1-w1-w2, 4))
                for w1 in W1_GRID for w2 in W2_GRID
                if 0.02 <= round(1-w1-w2, 4) <= 0.50]
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
                  f"-> {best_w_score:.3f}th  [{time.perf_counter()-t:.0f}s]")

    valid_c = [x for x in stage_c if x["mean_pct"] is not None]
    top10_c = sorted(valid_c, key=lambda x: x["mean_pct"], reverse=True)[:10]

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

    # Convergence check on weights
    at_w1_edge = (best_w1 == min(W1_GRID) or best_w1 == max(W1_GRID))
    converged  = (not at_radius_edge) and (not at_w1_edge)

    # ── Summary ───────────────────────────────────────────────
    total_time = time.perf_counter() - T_START
    best = {
        "radius_miles":   best_r,
        "beta":           best_b,
        "K":              best_k,
        "W1":             best_w1,
        "W2":             best_w2,
        "W3":             best_w3,
        "mean_pct_rank":  round(best_w_score, 3),
        "loocv_pct_rank": round(best_final_score, 3),
    }

    improved = best_final_score > SEARCH3_LOOCV

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

    print(f"\n  Progress across searches:")
    print(f"    search_1 (mean-pct)   : 60.63th pctile")
    print(f"    search_2 (coord.desc) : 60.958th pctile")
    print(f"    search_3 (LOOCV)      : {SEARCH3_LOOCV}th pctile")
    print(f"    search_4 (LOOCV ext.) : {best['loocv_pct_rank']}th pctile  "
          f"{'<- improved' if improved else '<- no improvement'}")

    if converged:
        print(f"\n  CONVERGED: winner is interior to grid on all dimensions.")
        print(f"  These are your final parameters.")
    else:
        if at_radius_edge:
            edge_dir = "below" if best_r == min(RADIUS_GRID) else "above"
            print(f"\n  NOT CONVERGED: radius still at grid edge ({edge_dir}).")
            print(f"  Consider extending radius grid further {edge_dir}.")
        if at_w1_edge:
            edge_dir = "above" if best_w1 == max(W1_GRID) else "below"
            print(f"  NOT CONVERGED: W1 still at grid edge ({edge_dir}).")

    # Top-10 final
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

    # ── Save results ─────────────────────────────────────────
    output = {
        "brand":        "Harris Teeter",
        "city":         "Charlotte, NC",
        "test":         "hyperparam_search_4_loocv_extended",
        "best_params":  best,
        "converged":    converged,
        "improved_over_search3": improved,
        "stage_a_top20": top20_a,
        "stage_b_loocv": stage_b,
        "stage_c_top10_weights": top10_c,
        "stage_d_loocv_weights": top10_final,
        "runtime_minutes": round(total_time / 60, 1),
    }

    json_path = os.path.join(RESULTS_DIR, "hyperparam_search_4_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    report = [
        "V5 HYPERPARAM SEARCH 4 -- Extended Grid + LOOCV -- Harris Teeter",
        "=" * 65,
        "",
        "WHY SEARCH 4:",
        "  Search 3 winner hit grid edges: radius=4.0 (bottom) and W1=0.70 (top)",
        "  Search 4 extends radius down to 2.5mi and W1 up to 0.85",
        "",
        f"FINAL RESULT  (converged={converged})",
        f"  radius={best_r}mi  beta={best_b}  K={best_k}",
        f"  W1={best_w1} (Huff demand)  W2={best_w2} (Dijkstra)  W3={best_w3} (Comp penalty)",
        f"  Mean pct rank : {best['mean_pct_rank']}th",
        f"  LOOCV pct rank: {best['loocv_pct_rank']}th",
        "",
        "PROGRESS",
        "  search_1 : 60.63th",
        "  search_2 : 60.958th",
        f"  search_3 : {SEARCH3_LOOCV}th",
        f"  search_4 : {best['loocv_pct_rank']}th  ({'improved' if improved else 'no improvement'})",
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

    txt_path = os.path.join(RESULTS_DIR, "hyperparam_search_4_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # Update optimal_params.json only if improved
    if improved:
        optimal = {
            "source":         "V5_hyperparam_search_4_loocv_extended",
            "radius_miles":   best_r,
            "beta":           best_b,
            "K":              best_k,
            "W1":             best_w1,
            "W2":             best_w2,
            "W3":             best_w3,
            "mean_pct_rank":  round(best_w_score, 3),
            "loocv_pct_rank": round(best_final_score, 3),
        }
        opt_path = os.path.join(RESULTS_DIR, "optimal_params.json")
        with open(opt_path, "w") as f:
            json.dump(optimal, f, indent=2)
        print(f"\n  Updated: {opt_path}  (improved over search_3)")
    else:
        print(f"\n  optimal_params.json NOT updated (search_3 still better)")

    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()

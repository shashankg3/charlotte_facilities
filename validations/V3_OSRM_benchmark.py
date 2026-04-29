# =============================================================
# V3_OSRM_benchmark.py
#
# Complete V3 Paper Comparison benchmark — ALL methods on PC.
#
# Replicates the routing methodology from each paper:
#   Paper 1 (Jin & Lu 2022):    Sequential Dijkstra (scipy)
#   Paper 2 (Kang et al. 2020): Parallel Dijkstra (multiprocessing)
#   Our method:                 Multi-source Dijkstra (scipy, single call)
#   Paper 3 (Horton et al. 2025): OSRM CH queries (localhost:5000)
#
# REQUIRES:
#   - OSRM server running on localhost:5000 (see README)
#   - All project data in data/ folder
#   - bg_dist_matrix.npy in cache/ (generated if missing)
#
# USAGE:
#   python V3_OSRM_benchmark.py
#
# OUTPUT:
#   Results_final/V3_results/v3_pc_benchmark.json
#   Results_final/V3_results/v3_pc_benchmark.txt
# =============================================================

import os
import sys
import time
import json
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
import requests
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_approach2 import (
    load_all, norm_weight, MILE_M, CRS_M, CRS_LL,
    CIRCUITY_FACTOR, ACCESS_ALPHA,
)

# =============================================================
# CONFIG
# =============================================================
RADIUS_MILES = 5.0
RADIUS_M     = RADIUS_MILES * MILE_M
SPEED_MPS    = 35 * MILE_M / 3600.0
OSRM_URL     = "http://localhost:5000"
RESULTS_DIR  = "Results_final/V3_results"

# OSRM preprocessing time (measured during docker setup)
# Extract + Contract = total CH preprocessing
OSRM_EXTRACT_S  = None   # Will be set from user input
OSRM_CONTRACT_S = 2546.4  # Measured: osrm-contract output


def spearmanr(x, y):
    """Spearman rank correlation without scipy.stats."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float('nan'), None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    rho = np.dot(rx, ry) / (denom + 1e-15)
    return float(rho), None


# =============================================================
# DATA LOADING
# =============================================================
def load_data():
    """Load all project data and prepare BG/candidate arrays."""
    print("[DATA] Loading project data...")
    t0 = time.perf_counter()
    raw = load_all()

    bg_m       = raw["bg_m"]           # GeoDataFrame in CRS_M
    cands_m    = raw["cands_m"]        # GeoDataFrame in CRS_M
    cands_ll   = raw["cands_ll"]       # GeoDataFrame in lat/lon
    road_csr   = raw["road_csr"]       # scipy CSR matrix
    road_kdtree= raw["road_kdtree"]    # cKDTree of road nodes
    bg_node_ids= raw["bg_node_ids"]    # array: BG -> nearest road node

    # BG in lat/lon (for OSRM queries)
    bg_ll = bg_m.to_crs(CRS_LL)

    # BG centroids (metres)
    bg_centroids_m = np.c_[bg_m.geometry.centroid.x.values,
                           bg_m.geometry.centroid.y.values]

    # Candidate centroids (metres)
    cand_xy_m = np.c_[cands_m.geometry.x.values,
                      cands_m.geometry.y.values]

    # BG centroids (lon, lat for OSRM)
    bg_ll_coords = np.c_[bg_ll.geometry.centroid.x.values,
                         bg_ll.geometry.centroid.y.values]

    # Candidate centroids (lon, lat for OSRM)
    cands_ll_coords = np.c_[cands_ll.geometry.centroid.x.values,
                            cands_ll.geometry.centroid.y.values]

    # Demand weights (column names from model_approach2 cache)
    bg_pop  = bg_m["population"].values.astype(float) if "population" in bg_m.columns else bg_m["pop"].values.astype(float)
    bg_inc  = bg_m["median_income"].values.astype(float) if "median_income" in bg_m.columns else bg_m["income"].values.astype(float)
    bg_dens = bg_m["pop_per_sqmi"].values.astype(float) if "pop_per_sqmi" in bg_m.columns else bg_m["density"].values.astype(float)
    B = 0.4 * norm_weight(bg_pop) + 0.3 * norm_weight(bg_inc) + 0.3 * norm_weight(bg_dens)

    # Snap candidates to road network
    cand_node_ids = road_kdtree.query(cand_xy_m)[1]

    # Catchment: which BGs are within radius of each candidate
    bg_cand_pairs = {}  # {cand_idx: [bg_indices within catchment]}
    for k in range(len(cand_xy_m)):
        dists = np.sqrt(np.sum((bg_centroids_m - cand_xy_m[k])**2, axis=1))
        in_catch = np.where(dists <= RADIUS_M)[0]
        if len(in_catch) > 0:
            bg_cand_pairs[k] = in_catch

    elapsed = time.perf_counter() - t0
    print(f"[DATA] Loaded: {len(bg_m)} BGs, {len(cands_ll)} candidates, "
          f"{road_csr.shape[0]:,} nodes")
    print(f"[DATA] Time: {elapsed:.1f}s")

    return {
        "bg_ll": bg_ll, "cands_ll": cands_ll,
        "road_csr": road_csr,
        "bg_centroids_m": bg_centroids_m, "cand_xy_m": cand_xy_m,
        "bg_ll_coords": bg_ll_coords, "cands_ll_coords": cands_ll_coords,
        "B": B, "bg_node_ids": bg_node_ids, "cand_node_ids": cand_node_ids,
        "bg_cand_pairs": bg_cand_pairs,
        "road_kdtree": road_kdtree,
    }


# =============================================================
# SCORING FUNCTION (shared by all methods)
# =============================================================
def compute_accessibility_scores(t_bar_array):
    """Convert weighted mean travel times to accessibility scores."""
    scores = np.exp(-ACCESS_ALPHA * t_bar_array)
    scores[~np.isfinite(t_bar_array)] = np.nan
    return scores


# =============================================================
# PAPER 1: STANDARD (SEQUENTIAL) DIJKSTRA
# Jin & Lu (2022) — ArcGIS OD Cost Matrix
# =============================================================
def run_paper1(data):
    """Sequential Dijkstra: one call per candidate."""
    print("\n" + "="*65)
    print("  PAPER 1: Standard Sequential Dijkstra")
    print("  Replicating: Jin & Lu (2022) — ArcGIS OD Cost Matrix")
    print("="*65)

    road_csr      = data["road_csr"]
    cand_node_ids = data["cand_node_ids"]
    bg_node_ids   = data["bg_node_ids"]
    bg_cand_pairs = data["bg_cand_pairs"]
    B             = data["B"]
    N_cands       = len(cand_node_ids)

    t_bar = np.full(N_cands, np.nan)
    t0 = time.perf_counter()
    milestone = max(1, N_cands // 10)

    for k in range(N_cands):
        if k not in bg_cand_pairs:
            continue

        cand_node = int(cand_node_ids[k])
        in_catch  = bg_cand_pairs[k]

        # One Dijkstra from this candidate
        dist_vec = scipy_dijkstra(
            road_csr, directed=False, indices=[cand_node],
            return_predecessors=False
        )[0]

        total_wt = 0.0
        total_w  = 0.0
        for i in in_catch:
            d = dist_vec[bg_node_ids[i]]
            if not np.isfinite(d) or d <= 0:
                continue
            t_min = (d / SPEED_MPS) / 60.0
            total_wt += B[i] * t_min
            total_w  += B[i]

        if total_w > 1e-9:
            t_bar[k] = total_wt / total_w

        if (k + 1) % milestone == 0:
            elapsed = time.perf_counter() - t0
            pct = 100 * (k + 1) / N_cands
            rate = (k + 1) / elapsed
            remaining = (N_cands - k - 1) / rate
            print(f"  {pct:5.0f}% | {k+1}/{N_cands} | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    total_time = time.perf_counter() - t0
    print(f"\n  DONE: {total_time:.1f}s ({total_time/60:.1f}min)")
    return t_bar, total_time


# =============================================================
# PAPER 2: PARALLEL DIJKSTRA
# Kang et al. (2020) — multiprocessing
# =============================================================
def _worker_dijkstra(args):
    """Worker function for parallel Dijkstra."""
    road_csr, cand_node, bg_node_ids, in_catch, B, speed_mps = args
    dist_vec = scipy_dijkstra(
        road_csr, directed=False, indices=[cand_node],
        return_predecessors=False
    )[0]

    total_wt = 0.0
    total_w  = 0.0
    for i in in_catch:
        d = dist_vec[bg_node_ids[i]]
        if not np.isfinite(d) or d <= 0:
            continue
        t_min = (d / speed_mps) / 60.0
        total_wt += B[i] * t_min
        total_w  += B[i]

    if total_w > 1e-9:
        return total_wt / total_w
    return np.nan


def run_paper2(data, n_cores=4):
    """Parallel Dijkstra across CPU cores."""
    print("\n" + "="*65)
    print(f"  PAPER 2: Parallel Dijkstra ({n_cores} cores)")
    print("  Replicating: Kang et al. (2020) — P-E2SFCA")
    print("="*65)

    road_csr      = data["road_csr"]
    cand_node_ids = data["cand_node_ids"]
    bg_node_ids   = data["bg_node_ids"]
    bg_cand_pairs = data["bg_cand_pairs"]
    B             = data["B"]
    N_cands       = len(cand_node_ids)

    # Build work items
    work = []
    valid_indices = []
    for k in range(N_cands):
        if k in bg_cand_pairs:
            work.append((
                road_csr, int(cand_node_ids[k]),
                bg_node_ids, bg_cand_pairs[k], B, SPEED_MPS
            ))
            valid_indices.append(k)

    print(f"  Candidates with catchment: {len(work)}")
    t0 = time.perf_counter()

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(_worker_dijkstra, work)

    t_bar = np.full(N_cands, np.nan)
    for idx, val in zip(valid_indices, results):
        t_bar[idx] = val

    total_time = time.perf_counter() - t0
    print(f"\n  DONE: {total_time:.1f}s ({total_time/60:.1f}min)")
    return t_bar, total_time


# =============================================================
# OUR METHOD: MULTI-SOURCE DIJKSTRA
# Single scipy call from all BG sources
# =============================================================
def run_ours(data):
    """Multi-source Dijkstra: single compiled C call."""
    print("\n" + "="*65)
    print("  OUR METHOD: Multi-source Dijkstra")
    print("  scipy.sparse.csgraph.dijkstra — single compiled C call")
    print("="*65)

    road_csr      = data["road_csr"]
    cand_node_ids = data["cand_node_ids"]
    bg_node_ids   = data["bg_node_ids"]
    bg_cand_pairs = data["bg_cand_pairs"]
    B             = data["B"]
    N_cands       = len(cand_node_ids)

    unique_bg_nodes = np.unique(bg_node_ids)

    # Phase 1: Preprocessing (build distance matrix)
    print(f"  Phase 1: Building distance matrix "
          f"[{len(unique_bg_nodes)} x {road_csr.shape[0]:,}]")
    t_pre = time.perf_counter()

    dist_matrix = scipy_dijkstra(
        road_csr, directed=False,
        indices=unique_bg_nodes,
        return_predecessors=False,
    )

    preproc_time = time.perf_counter() - t_pre
    print(f"  Preprocessing: {preproc_time:.1f}s")

    # Save for other uses
    cache_path = "cache/bg_dist_matrix.npy"
    os.makedirs("cache", exist_ok=True)
    np.save(cache_path, dist_matrix)

    # Phase 2: Scoring (column lookup)
    node_to_row = {int(nid): r for r, nid in enumerate(unique_bg_nodes)}

    t_score = time.perf_counter()
    t_bar = np.full(N_cands, np.nan)

    for k in range(N_cands):
        if k not in bg_cand_pairs:
            continue

        cand_node = int(cand_node_ids[k])
        in_catch  = bg_cand_pairs[k]

        total_wt = 0.0
        total_w  = 0.0
        for i in in_catch:
            row = node_to_row.get(int(bg_node_ids[i]))
            if row is None:
                continue
            d = dist_matrix[row, cand_node]
            if not np.isfinite(d) or d <= 0:
                continue
            t_min = (d / SPEED_MPS) / 60.0
            total_wt += B[i] * t_min
            total_w  += B[i]

        if total_w > 1e-9:
            t_bar[k] = total_wt / total_w

    score_time = time.perf_counter() - t_score
    total_time = preproc_time + score_time
    print(f"  Scoring: {score_time:.1f}s")
    print(f"\n  DONE: {total_time:.1f}s (preproc={preproc_time:.1f}s + "
          f"scoring={score_time:.1f}s)")

    return t_bar, preproc_time, score_time


# =============================================================
# PAPER 3: OSRM CONTRACTION HIERARCHIES
# Horton et al. (2025) — OSRM routing backend
# =============================================================
def run_paper3_osrm(data, batch_size=100):
    """
    Query OSRM Table API for all candidate-BG distance pairs.

    Uses the /table/v1/driving endpoint, which computes a
    distance matrix using CH internally — exactly how Horton et al.
    would have used OSRM.

    We batch BG sources to avoid URL length limits.
    """
    print("\n" + "="*65)
    print("  PAPER 3: OSRM Contraction Hierarchies")
    print("  Replicating: Horton et al. (2025) — OSRM routing backend")
    print("="*65)

    bg_ll_coords    = data["bg_ll_coords"]     # [lon, lat]
    cands_ll_coords = data["cands_ll_coords"]  # [lon, lat]
    bg_cand_pairs   = data["bg_cand_pairs"]
    B               = data["B"]
    N_cands         = len(cands_ll_coords)
    N_bgs           = len(bg_ll_coords)

    # Test OSRM connectivity
    try:
        r = requests.get(f"{OSRM_URL}/route/v1/driving/"
                        f"{bg_ll_coords[0,0]},{bg_ll_coords[0,1]};"
                        f"{cands_ll_coords[0,0]},{cands_ll_coords[0,1]}"
                        f"?overview=false", timeout=5)
        if r.status_code != 200:
            print(f"  ERROR: OSRM not responding (status {r.status_code})")
            return None, 0.0
        print(f"  OSRM server: OK")
    except Exception as e:
        print(f"  ERROR: Cannot connect to OSRM: {e}")
        print(f"  Make sure OSRM is running: docker run -t -d -p 5000:5000 ...")
        return None, 0.0

    # Strategy: Use OSRM /table/v1 API
    # Send all BGs as sources, candidates as destinations in batches
    # The table API returns a duration matrix (seconds)

    print(f"  Querying {N_bgs} BGs x {N_cands} candidates via OSRM Table API")
    print(f"  Batch size: {batch_size} candidates per request")

    # Build all coordinates: BGs first, then candidates
    # OSRM table API: coordinates = "lon,lat;lon,lat;..."
    # sources = BG indices, destinations = candidate indices

    t0 = time.perf_counter()

    # We'll query BG-by-BG using route API for each candidate's catchment BGs
    # This is more representative of how Horton et al. used OSRM:
    # they queried travel times for specific OD pairs

    # Actually, the most efficient OSRM approach is the Table API
    # which computes a full NxM matrix in one call.
    # But with 559 BGs x 2450 candidates = too many coordinates for one call.
    # OSRM default max is ~100 coordinates per request.

    # Strategy: For each candidate, query distances to its catchment BGs
    # using the Table API in small batches.

    t_bar = np.full(N_cands, np.nan)
    total_queries = 0
    milestone = max(1, N_cands // 10)

    for k in range(N_cands):
        if k not in bg_cand_pairs:
            continue

        in_catch = bg_cand_pairs[k]
        cand_lon, cand_lat = cands_ll_coords[k]

        # Build coordinate string: candidate + catchment BGs
        coords = f"{cand_lon},{cand_lat}"
        for i in in_catch:
            bg_lon, bg_lat = bg_ll_coords[i]
            coords += f";{bg_lon},{bg_lat}"

        # Table API: source=0 (candidate), destinations=1..N (BGs)
        dest_indices = ";".join(str(j+1) for j in range(len(in_catch)))
        url = (f"{OSRM_URL}/table/v1/driving/{coords}"
               f"?sources=0&destinations={dest_indices}"
               f"&annotations=duration")

        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            result = r.json()
            if result.get("code") != "Ok":
                continue

            durations = result["durations"][0]  # from candidate to all BGs
            total_queries += 1

            total_wt = 0.0
            total_w  = 0.0
            for j, i in enumerate(in_catch):
                dur = durations[j]
                if dur is None or dur <= 0:
                    continue
                t_min = dur / 60.0  # OSRM returns seconds
                total_wt += B[i] * t_min
                total_w  += B[i]

            if total_w > 1e-9:
                t_bar[k] = total_wt / total_w

        except requests.exceptions.RequestException:
            continue

        if (k + 1) % milestone == 0:
            elapsed = time.perf_counter() - t0
            pct = 100 * (k + 1) / N_cands
            rate = (k + 1) / elapsed
            remaining = (N_cands - k - 1) / rate
            print(f"  {pct:5.0f}% | {k+1}/{N_cands} | "
                  f"{total_queries} queries | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    query_time = time.perf_counter() - t0
    print(f"\n  DONE: {query_time:.1f}s ({query_time/60:.1f}min)")
    print(f"  Total OSRM queries: {total_queries}")

    return t_bar, query_time


# =============================================================
# MAIN
# =============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  V3: PAPER COMPARISON BENCHMARK (PC)")
    print("  All methods on same hardware for fair comparison")
    print("=" * 65)

    # Load data once
    data = load_data()
    N_bgs   = len(data["bg_ll"])
    N_cands = len(data["cands_ll"])
    N_nodes = data["road_csr"].shape[0]
    print(f"\n  Charlotte: {N_bgs} BGs, {N_cands} candidates, "
          f"{N_nodes:,} nodes\n")

    results = {}

    # ---- Paper 1: Standard Dijkstra ----
    t_bar_p1, time_p1 = run_paper1(data)
    results["paper1"] = {
        "label": "Standard Dijkstra (sequential)",
        "reference": "Jin & Lu (2022)",
        "preprocess_s": 0,
        "query_s": round(time_p1, 1),
        "total_s": round(time_p1, 1),
    }

    # ---- Paper 2: Parallel Dijkstra ----
    n_cores = min(4, mp.cpu_count())
    t_bar_p2, time_p2 = run_paper2(data, n_cores=n_cores)
    results["paper2"] = {
        "label": f"Parallel Dijkstra ({n_cores} cores)",
        "reference": "Kang et al. (2020)",
        "preprocess_s": 0,
        "query_s": round(time_p2, 1),
        "total_s": round(time_p2, 1),
    }

    # ---- Our Method: Multi-source Dijkstra ----
    t_bar_ours, preproc_ours, score_ours = run_ours(data)
    total_ours = preproc_ours + score_ours
    results["ours"] = {
        "label": "Multi-source Dijkstra (ours)",
        "reference": "This study",
        "preprocess_s": round(preproc_ours, 1),
        "query_s": round(score_ours, 1),
        "total_s": round(total_ours, 1),
    }

    # ---- Paper 3: OSRM CH ----
    t_bar_p3, time_p3 = run_paper3_osrm(data)
    results["paper3"] = {
        "label": "OSRM CH (Contraction Hierarchies)",
        "reference": "Horton et al. (2025)",
        "preprocess_s": round(OSRM_CONTRACT_S, 1),
        "query_s": round(time_p3, 1),
        "total_s": round(OSRM_CONTRACT_S + time_p3, 1),
        "note": "Preprocessing = osrm-contract (measured separately via Docker)",
    }

    # ---- Ranking Comparisons (Spearman rho vs Paper 1 baseline) ----
    scores_p1   = compute_accessibility_scores(t_bar_p1)
    scores_p2   = compute_accessibility_scores(t_bar_p2)
    scores_ours = compute_accessibility_scores(t_bar_ours)

    rho_p2, _   = spearmanr(scores_p1, scores_p2)
    rho_ours, _ = spearmanr(scores_p1, scores_ours)

    results["paper1"]["spearman_rho"] = "baseline"
    results["paper2"]["spearman_rho"] = round(rho_p2, 4)
    results["ours"]["spearman_rho"]   = round(rho_ours, 4)

    if t_bar_p3 is not None:
        scores_p3 = compute_accessibility_scores(t_bar_p3)
        rho_p3, _ = spearmanr(scores_p1, scores_p3)
        results["paper3"]["spearman_rho"] = round(rho_p3, 4)

    # ---- Speedups vs Paper 1 ----
    results["paper2"]["speedup_vs_p1"] = round(time_p1 / time_p2, 2)
    results["ours"]["speedup_vs_p1"]   = round(time_p1 / total_ours, 2)
    if time_p3 > 0:
        results["paper3"]["speedup_vs_p1"] = round(time_p1 / (OSRM_CONTRACT_S + time_p3), 2)

    # ---- Print Summary ----
    print("\n" + "="*65)
    print("  V3 RESULTS SUMMARY")
    print("="*65)
    print(f"\n  Charlotte: {N_bgs} BGs, {N_cands} candidates, {N_nodes:,} nodes")
    print(f"\n  {'Method':<35} {'Preproc':>10} {'Query':>10} {'Total':>10} {'Spearman':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for key in ["paper1", "paper2", "ours", "paper3"]:
        r = results[key]
        pre = f"{r['preprocess_s']:.1f}s" if r['preprocess_s'] > 0 else "none"
        qry = f"{r['query_s']:.1f}s"
        tot = f"{r['total_s']:.1f}s"
        rho = str(r.get('spearman_rho', 'N/A'))
        print(f"  {r['label']:<35} {pre:>10} {qry:>10} {tot:>10} {rho:>10}")

    print(f"\n  Speedup (ours vs Paper 1): {results['ours']['speedup_vs_p1']}x")
    print(f"  Speedup (ours vs Paper 2): {round(time_p2 / total_ours, 2)}x")

    # ---- Save ----
    output = {
        "benchmark": "V3_paper_comparison_PC",
        "hardware": {
            "cpu": "Intel Core i5-1035G1 @ 1.00GHz",
            "cores": 4,
            "threads": 8,
            "ram_gb": 8,
            "os": "Windows",
            "docker": "29.0.1",
            "osrm": "osrm-backend:latest (Docker)",
        },
        "graph": {
            "city": "Charlotte, NC",
            "nodes": N_nodes,
            "bgs": N_bgs,
            "candidates": N_cands,
        },
        "results": results,
    }

    json_path = os.path.join(RESULTS_DIR, "v3_pc_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Text summary
    txt_path = os.path.join(RESULTS_DIR, "v3_pc_benchmark.txt")
    with open(txt_path, "w") as f:
        f.write("V3 Paper Comparison Benchmark (PC)\n")
        f.write(f"Hardware: Intel i5-1035G1, 8GB RAM, Windows, Docker 29.0.1\n")
        f.write(f"Graph: Charlotte, {N_nodes:,} nodes, {N_bgs} BGs, {N_cands} cands\n\n")
        for key in ["paper1", "paper2", "ours", "paper3"]:
            r = results[key]
            f.write(f"{r['label']}: {r['total_s']:.1f}s total "
                    f"(preproc={r['preprocess_s']:.1f}s, query={r['query_s']:.1f}s)\n")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()

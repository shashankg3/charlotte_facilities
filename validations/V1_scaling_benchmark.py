"""
================================================================================
benchmark_scaling_hpc.py  --  HPC-Ready Multi-Source Dijkstra Scaling Benchmark
================================================================================

PURPOSE
-------
Fair, reproducible timing of Multi-Source Dijkstra (our method) vs Standard
(naive, one-source-at-a-time) Dijkstra across three city scales:
  Charlotte  --  282,751 road nodes
  Atlanta    --  916,452 road nodes
  Los Angeles-- 1,018,083 road nodes

FAIR MEASUREMENT GUARANTEE
---------------------------
Pass --ram-gb <TOTAL_NODE_RAM> so EVERY city uses the SAME RAM budget and the
same chunking strategy.  On a >= 64 GB HPC node all three cities run in a
SINGLE chunk (no chunk overhead) --> clean, comparable numbers.

  Charlotte : 559 BGs  x 282k nodes x 8B =  1.3 GB  (fits in 64 GB easily)
  Atlanta   : 1148 cands x 916k nodes x 8B =  8.4 GB  (fits)
  LA        : 6400 BGs  x 1018k nodes x 8B = 52.0 GB  (fits in 64 GB)

If a city still needs chunking at your RAM size, the script clearly flags it
as NON-FAIR so you know the result includes chunk overhead.

USAGE
-----
  # Run all cities, tell script node has 80 GB
  python benchmark_scaling_hpc.py --all --ram-gb 80

  # Single city
  python benchmark_scaling_hpc.py --city Charlotte --ram-gb 80

  # Skip standard Dijkstra (saves hours for LA) -- use extrapolation instead
  python benchmark_scaling_hpc.py --all --ram-gb 80 --skip-std

  # Parallel standard Dijkstra (Linux HPC -- uses fork)
  python benchmark_scaling_hpc.py --all --ram-gb 80 --workers 16

  # Preview memory plan without running
  python benchmark_scaling_hpc.py --dry-run --ram-gb 80

  # Reprint table from saved JSON results (no re-run)
  python benchmark_scaling_hpc.py --table-only

HPC SLURM EXAMPLE
------------------
  #!/bin/bash
  #SBATCH --job-name=dijkstra_scaling
  #SBATCH --mem=80G
  #SBATCH --cpus-per-task=16
  #SBATCH --time=04:00:00
  #SBATCH --output=scaling_%j.log
  module load python/3.10
  python benchmark_scaling_hpc.py --all --workers 16
  (script auto-reads SLURM_MEM_PER_NODE and SLURM_CPUS_PER_TASK)

DATA FILES NEEDED  (place in data/ folder)
------------------------------------------
  data/charlotte_roads_drive.geojson   (already present)
  data/atlanta_roads_drive.geojson     (already present)
  data/la_roads_drive.geojson          (already present)
  data/candidates_osm.geojson          (Charlotte, 2,450 candidates)
  data/candidates_atlanta.geojson      (Atlanta,   1,148 candidates)
  data/candidates_la.geojson           (LA,       14,208 candidates)

REQUIREMENTS
------------
  pip install scipy numpy geopandas shapely networkx pyogrio
  pip install psutil            # recommended for RAM detection
  pip install joblib            # optional, for --workers on Windows
================================================================================
"""

import os
import sys
import time
import json
import gc
import argparse
import warnings
import math
from datetime import datetime

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.csgraph import dijkstra as csr_dijkstra

warnings.filterwarnings("ignore")

# ============================================================
# DIRECTORIES
# ============================================================
CACHE_DIR  = "benchmark_cache"
OUTPUT_DIR = "results"
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CITY CONFIGURATIONS
# ============================================================
CITIES = {
    "Charlotte": {
        "local_roads_file": "data/charlotte_roads_drive.geojson",
        "osmnx_query":      "Mecklenburg County, North Carolina, USA",
        "candidates_file":  "data/candidates_osm.geojson",
        "n_block_groups":   559,   # Mecklenburg BGs clipped to Charlotte boundary
        "crs_m":            32119, # NC State Plane metres
    },
    "Atlanta": {
        "local_roads_file": "data/atlanta_roads_drive.geojson",
        "osmnx_query": [
            "Fulton County, Georgia, USA",
            "DeKalb County, Georgia, USA",
            "Cobb County, Georgia, USA",
            "Gwinnett County, Georgia, USA",
            "Clayton County, Georgia, USA",
        ],
        "candidates_file":  "data/candidates_atlanta.geojson",
        "n_block_groups":   1800,
        "crs_m":            32617, # UTM 17N metres
    },
    "Los Angeles": {
        "local_roads_file": "data/la_roads_drive.geojson",
        "osmnx_query":      "Los Angeles County, California, USA",
        "candidates_file":  "data/candidates_la.geojson",
        "n_block_groups":   6400,
        "crs_m":            32611, # UTM 11N metres
    },
}

# ============================================================
# HELPERS
# ============================================================
def ts():
    """Timestamp prefix for every log line -- essential for HPC log files."""
    return datetime.now().strftime("[%H:%M:%S]")


def log(msg):
    print(f"{ts()} {msg}", flush=True)


def detect_ram_gb(explicit_gb=None):
    """
    Return usable RAM in GB.
    Priority: --ram-gb flag > SLURM env > psutil > /proc/meminfo > fallback 8 GB.
    We reserve 15% headroom so the OS and other processes don't OOM us.
    """
    HEADROOM = 0.85   # use 85% of reported RAM

    if explicit_gb is not None:
        gb = float(explicit_gb) * HEADROOM
        log(f"RAM budget: {explicit_gb} GB specified, using {gb:.1f} GB (85% headroom)")
        return gb

    # SLURM allocates memory via SLURM_MEM_PER_NODE (in MB)
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        try:
            gb = float(slurm_mem) / 1024.0 * HEADROOM
            log(f"RAM budget: SLURM_MEM_PER_NODE={slurm_mem} MB -> {gb:.1f} GB (85%)")
            return gb
        except ValueError:
            pass

    # psutil (most reliable on local machines)
    try:
        import psutil
        gb = psutil.virtual_memory().available / (1024 ** 3) * HEADROOM
        log(f"RAM budget: psutil available {gb/HEADROOM:.1f} GB, using {gb:.1f} GB (85%)")
        return gb
    except ImportError:
        pass

    # /proc/meminfo (Linux without psutil)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    gb = int(line.split()[1]) / (1024 ** 2) * HEADROOM
                    log(f"RAM budget: /proc/meminfo available {gb/HEADROOM:.1f} GB, "
                        f"using {gb:.1f} GB (85%)")
                    return gb
    except Exception:
        pass

    log("RAM budget: could not detect -- defaulting to 8 GB. "
        "Use --ram-gb to set explicitly.")
    return 8.0


def detect_workers(explicit_workers=None):
    """Return number of parallel workers for standard Dijkstra."""
    if explicit_workers is not None:
        return int(explicit_workers)
    # SLURM CPU allocation
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        w = int(slurm_cpus)
        log(f"Workers: SLURM_CPUS_PER_TASK={w}")
        return w
    return 1


# ============================================================
# STEP 1  --  Load road GeoJSON
# ============================================================
def _count_vertices(geometry_series):
    total = 0
    for g in geometry_series:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "MultiLineString":
            for part in g.geoms:
                total += len(list(part.coords))
        elif g.geom_type == "LineString":
            total += len(list(g.coords))
    return total


def load_roads(city_name, config):
    """
    Load road edges as GeoDataFrame of LineStrings.
    Uses local file if present, otherwise downloads via osmnx.
    """
    import geopandas as gpd

    local_file = config.get("local_roads_file", "")
    if local_file and os.path.exists(local_file):
        log(f"  Road file: {local_file} (local)")
        t0 = time.perf_counter()
        edges = gpd.read_file(local_file, engine="pyogrio")
        if edges.crs is None:
            edges.set_crs("EPSG:4326", inplace=True)
        edges = edges.to_crs("EPSG:4326")
        verts = _count_vertices(edges.geometry)
        log(f"  {len(edges):,} segments | {verts:,} vertices | loaded in "
            f"{time.perf_counter()-t0:.1f}s")
        return edges

    # osmnx download fallback
    cache_file = os.path.join(OUTPUT_DIR, f"{city_name}_roads_drive.geojson")
    if os.path.exists(cache_file):
        log(f"  Road file: {cache_file} (cached osmnx export)")
        edges = gpd.read_file(cache_file, engine="pyogrio")
        verts = _count_vertices(edges.geometry)
        log(f"  {len(edges):,} segments | {verts:,} vertices")
        return edges

    import osmnx as ox
    query = config["osmnx_query"]
    log(f"  Downloading from OSM: {query}")
    if isinstance(query, list):
        graphs = []
        for i, q in enumerate(query):
            log(f"    [{i+1}/{len(query)}] {q} ...")
            g = ox.graph_from_place(q, network_type="drive")
            log(f"    -> {g.number_of_nodes():,} nodes")
            graphs.append(g)
        G = nx.compose_all(graphs)
    else:
        G = ox.graph_from_place(query, network_type="drive")
    log(f"  osmnx: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]].copy()
    edges = edges.to_crs(4326)
    del G; gc.collect()

    log(f"  Saving -> {cache_file} ...")
    edges.to_file(cache_file, driver="GeoJSON")
    verts = _count_vertices(edges.geometry)
    log(f"  {len(edges):,} segments | {verts:,} vertices")
    return edges


# ============================================================
# STEP 2  --  Build vertex graph (exact copy of model_approach2)
# ============================================================
def _build_graph_from_roads(roads_m, snap_tol_m=1.0):
    """
    Every vertex in every LineString becomes a graph node.
    Exact same logic as model_approach2.py -- ensures benchmark matches
    the live application graph.
    """
    if roads_m is None or roads_m.empty:
        return None, None

    def _snap(x, y):
        return (round(x / snap_tol_m) * snap_tol_m,
                round(y / snap_tol_m) * snap_tol_m)

    G = nx.Graph()
    node_coords = {}

    for geom in roads_m.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            x_prev, y_prev = coords[0]
            u = _snap(x_prev, y_prev)
            node_coords.setdefault(u, u)
            for (x, y) in coords[1:]:
                v = _snap(x, y)
                if v != u:
                    dist = float(np.hypot(v[0] - u[0], v[1] - u[1]))
                    if np.isfinite(dist) and dist > 0:
                        if G.has_edge(u, v):
                            if dist < G[u][v]["weight"]:
                                G[u][v]["weight"] = dist
                        else:
                            G.add_edge(u, v, weight=dist)
                    node_coords.setdefault(v, v)
                    u = v

    return (G, node_coords) if G.number_of_nodes() > 0 else (None, None)


def graph_to_csr(G, node_coords):
    """Convert NetworkX graph to scipy CSR sparse matrix."""
    node_ids = list(node_coords.keys())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    u_list, v_list, w_list = [], [], []
    for u, v, data in G.edges(data=True):
        ui, vi = node_to_idx[u], node_to_idx[v]
        w = float(data.get("weight", 0.0))
        if w <= 0:
            continue
        u_list += [ui, vi]
        v_list += [vi, ui]
        w_list += [w,  w ]

    csr = csr_matrix(
        (np.array(w_list, dtype=np.float64),
         (np.array(u_list, dtype=np.int32),
          np.array(v_list, dtype=np.int32))),
        shape=(n, n)
    )
    return csr, n


# ============================================================
# STEP 3  --  Load / build CSR (with caching)
# ============================================================
def get_csr(city_name, config, force_rebuild=False):
    """
    Return (csr, n_nodes).  Loads from benchmark_cache/ if available,
    otherwise runs Steps 1 & 2 and caches the result.
    """
    csr_file   = os.path.join(CACHE_DIR, f"{city_name}_csr.npz")
    nodes_file = os.path.join(CACHE_DIR, f"{city_name}_nodes.npy")

    if not force_rebuild and os.path.exists(csr_file) and os.path.exists(nodes_file):
        log(f"  [cache HIT] Loading CSR from {CACHE_DIR}/")
        t0  = time.perf_counter()
        csr = load_npz(csr_file)
        n   = int(csr.shape[0])
        log(f"  {n:,} nodes | {csr.nnz:,} edges | {time.perf_counter()-t0:.1f}s")
        return csr, n

    # Build from scratch
    log(f"\n[Step 1] Load road GeoJSON")
    edges_ll = load_roads(city_name, config)

    log(f"\n[Step 2] Project -> build vertex graph -> CSR")
    crs_m   = config["crs_m"]
    t0      = time.perf_counter()
    edges_m = edges_ll.to_crs(f"EPSG:{crs_m}")
    del edges_ll; gc.collect()

    G, node_coords = _build_graph_from_roads(edges_m)
    del edges_m; gc.collect()

    if G is None:
        raise RuntimeError(f"No graph built for {city_name}")

    log(f"  Vertex graph: {G.number_of_nodes():,} nodes | "
        f"{G.number_of_edges():,} edges | {time.perf_counter()-t0:.1f}s")

    log(f"  Building CSR ...")
    csr, n = graph_to_csr(G, node_coords)
    node_xy = np.array(list(node_coords.values()), dtype=np.float32)
    del G, node_coords; gc.collect()

    log(f"  Saving CSR cache ...")
    save_npz(csr_file,   csr)
    np.save(nodes_file, node_xy)
    log(f"  {csr_file}  ({os.path.getsize(csr_file)/1e6:.1f} MB)")

    return csr, n


# ============================================================
# STEP 4  --  Memory plan  (call before any Dijkstra)
# ============================================================
def memory_plan(cities_configs, ram_gb):
    """
    Print a table showing chunk strategy for each city given the RAM budget.
    Flags cities that still need chunking as NON-FAIR.
    Returns dict city -> chunk_size.
    """
    log("")
    log("=" * 70)
    log("  MEMORY PLAN  (same RAM budget for ALL cities = fair comparison)")
    log(f"  Available RAM budget: {ram_gb:.1f} GB")
    log("=" * 70)
    log(f"  {'City':<14} {'N_src':>6} {'N_nodes':>10} "
        f"{'Matrix GB':>10} {'Chunks':>7}  {'Fair?':>6}")
    log(f"  {'-'*60}")

    plan = {}
    for city, cfg in cities_configs.items():
        csr_file = os.path.join(CACHE_DIR, f"{city}_csr.npz")
        if os.path.exists(csr_file):
            n_nodes = int(load_npz(csr_file).shape[0])
        else:
            n_nodes = 0   # unknown until built

        n_src        = cfg["n_block_groups"]
        matrix_gb    = n_src * max(n_nodes, 1) * 8 / (1024 ** 3)
        bytes_per_src = max(n_nodes, 1) * 8
        chunk_size   = max(1, int(ram_gb * (1024 ** 3) / bytes_per_src))
        n_chunks     = max(1, math.ceil(n_src / chunk_size))
        fair         = "YES" if n_chunks == 1 else f"NO ({n_chunks} chunks)"

        log(f"  {city:<14} {n_src:>6,} {n_nodes:>10,} "
            f"{matrix_gb:>10.2f} {n_chunks:>7}  {fair:>6}")
        plan[city] = {"chunk_size": chunk_size, "n_chunks": n_chunks,
                      "matrix_gb": matrix_gb, "fair": n_chunks == 1}

    log(f"  {'-'*60}")
    log("  NOTE: Cities with Chunks > 1 are non-fair due to chunk overhead.")
    log("        Use a larger --ram-gb value to eliminate chunking.")
    log("")
    return plan


# ============================================================
# STEP 5  --  Multi-Source Dijkstra  (OUR METHOD)
# ============================================================
def run_multisource_dijkstra(csr, n_nodes, n_sources, ram_gb, city_name):
    """
    Run multi-source Dijkstra from `n_sources` random source nodes.
    Chunk size is determined by the SHARED ram_gb budget.
    Returns dict with timing results.
    """
    rng     = np.random.default_rng(seed=42)
    sources = rng.choice(n_nodes, size=min(n_sources, n_nodes),
                         replace=False).astype(np.int32)

    bytes_per_src = n_nodes * 8
    chunk_size    = max(1, int(ram_gb * (1024 ** 3) / bytes_per_src))
    n_chunks      = max(1, math.ceil(len(sources) / chunk_size))
    matrix_gb     = len(sources) * n_nodes * 8 / (1024 ** 3)
    fair          = n_chunks == 1

    log(f"")
    log(f"  --- Multi-Source Dijkstra (OUR METHOD) ---")
    log(f"  Sources  : {len(sources):,}  (BG centroids)")
    log(f"  Nodes    : {n_nodes:,}")
    log(f"  RAM budget: {ram_gb:.1f} GB  |  Matrix: {matrix_gb:.2f} GB")
    if n_chunks == 1:
        log(f"  Strategy : SINGLE RUN -- all sources at once (clean, no overhead)")
    else:
        per_chunk_gb = chunk_size * n_nodes * 8 / (1024 ** 3)
        log(f"  Strategy : {n_chunks} chunks x ~{chunk_size} sources "
            f"(~{per_chunk_gb:.1f} GB/chunk)")
        log(f"  WARNING  : Chunked run -- chunk overhead included in time.")
        log(f"             Increase --ram-gb for a single-run fair result.")

    total_time = 0.0
    source_chunks = np.array_split(sources, n_chunks)
    for i, chunk in enumerate(source_chunks):
        if n_chunks > 1:
            log(f"  Chunk {i+1}/{n_chunks} ({len(chunk)} sources) ...")
        t0          = time.perf_counter()
        dist_matrix = csr_dijkstra(csr, directed=False,
                                   indices=chunk, return_predecessors=False)
        chunk_time  = time.perf_counter() - t0
        total_time += chunk_time
        if n_chunks > 1:
            log(f"    -> {chunk_time:.2f}s")
        del dist_matrix; gc.collect()

    log(f"  TOTAL    : {_fmt_time(total_time)}  "
        f"({'FAIR - single run' if fair else f'NON-FAIR - {n_chunks} chunks'})")

    return {
        "multisource_time_s": round(total_time, 3),
        "multisource_n_sources": int(len(sources)),
        "multisource_n_chunks": int(n_chunks),
        "multisource_fair": fair,
        "matrix_gb": round(matrix_gb, 3),
        "ram_budget_gb": round(ram_gb, 1),
    }


# ============================================================
# STEP 6  --  Standard (naive) Dijkstra  (COMPARISON BASELINE)
# ============================================================

# --- Worker used for parallel mode (Linux fork -- no serialisation cost) ---
_GLOBAL_CSR = None

def _init_worker(csr_data, indices, indptr, shape):
    global _GLOBAL_CSR
    from scipy.sparse import csr_matrix as _csr
    _GLOBAL_CSR = _csr((csr_data, indices, indptr), shape=shape)

def _run_one(src):
    d = csr_dijkstra(_GLOBAL_CSR, directed=False,
                     indices=int(src), return_predecessors=False)
    del d
    return 1


def run_standard_dijkstra(csr, n_nodes, n_candidates, workers=1,
                          skip=False, extrapolate_from=100):
    """
    Run one-source-at-a-time Dijkstra for all n_candidates.
    Returns dict with timing results.

    skip=True: measure first `extrapolate_from` runs, then extrapolate to full.
    workers>1: parallel execution (works efficiently on Linux HPC via fork).
    """
    rng     = np.random.default_rng(seed=99)
    sources = rng.choice(n_nodes, size=min(n_candidates, n_nodes),
                         replace=False).astype(np.int32)

    log(f"")
    log(f"  --- Standard (Naive) Dijkstra (BASELINE) ---")
    log(f"  Sources: {len(sources):,} candidates | Workers: {workers}")

    if skip:
        # Measure first `extrapolate_from` runs, extrapolate to full
        sample = sources[:min(extrapolate_from, len(sources))]
        log(f"  Mode: EXTRAPOLATE from first {len(sample)} runs "
            f"(--skip-std flag set)")
        t0 = time.perf_counter()
        for src in sample:
            d = csr_dijkstra(csr, directed=False,
                             indices=int(src), return_predecessors=False)
            del d
        sample_time = time.perf_counter() - t0
        per_run     = sample_time / len(sample)
        estimated   = per_run * len(sources)
        log(f"  Sample: {len(sample)} runs in {sample_time:.2f}s  "
            f"({per_run:.4f}s/run)")
        log(f"  EXTRAPOLATED total: {_fmt_time(estimated)}  "
            f"(NOT directly measured)")
        return {
            "std_time_s":           round(estimated, 2),
            "std_is_extrapolated":  True,
            "std_sample_runs":      int(len(sample)),
            "std_per_run_s":        round(per_run, 5),
            "std_n_candidates":     int(len(sources)),
        }

    # Full measurement
    report_every = max(1, len(sources) // 20)

    if workers > 1 and sys.platform != "win32":
        # Parallel on Linux/Mac via fork (CSR shared, no serialisation)
        from concurrent.futures import ProcessPoolExecutor
        log(f"  Mode: PARALLEL ({workers} workers, fork)")
        t0   = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(csr.data, csr.indices, csr.indptr, csr.shape)
        ) as pool:
            for done, _ in enumerate(pool.map(_run_one, sources.tolist(),
                                              chunksize=max(1, len(sources)//workers//4))):
                if (done + 1) % report_every == 0:
                    elapsed = time.perf_counter() - t0
                    eta     = elapsed / (done + 1) * (len(sources) - done - 1)
                    log(f"  Progress {done+1:,}/{len(sources):,} "
                        f"({100*(done+1)/len(sources):.0f}%) "
                        f"-- {elapsed:.1f}s elapsed -- ETA ~{eta:.0f}s")
        std_time = time.perf_counter() - t0

    else:
        # Sequential (Windows or workers=1)
        if workers > 1:
            log(f"  Note: parallel workers not supported on Windows; "
                f"running sequential.")
        log(f"  Mode: SEQUENTIAL")
        t0 = time.perf_counter()
        for i, src in enumerate(sources):
            d = csr_dijkstra(csr, directed=False,
                             indices=int(src), return_predecessors=False)
            del d
            if (i + 1) % report_every == 0:
                elapsed = time.perf_counter() - t0
                eta     = elapsed / (i + 1) * (len(sources) - i - 1)
                log(f"  Progress {i+1:,}/{len(sources):,} "
                    f"({100*(i+1)/len(sources):.0f}%) "
                    f"-- {elapsed:.1f}s elapsed -- ETA ~{eta:.0f}s")
        std_time = time.perf_counter() - t0

    per_run = std_time / len(sources)
    log(f"  FULLY MEASURED: {len(sources):,} runs in {_fmt_time(std_time)} "
        f"({per_run:.4f}s/run)")

    return {
        "std_time_s":           round(std_time, 3),
        "std_is_extrapolated":  False,
        "std_per_run_s":        round(per_run, 5),
        "std_n_candidates":     int(len(sources)),
    }


# ============================================================
# CORE: run one city end-to-end
# ============================================================
def run_city(city_name, config, ram_gb, workers=1,
             skip_std=False, force_rebuild=False):
    import geopandas as gpd

    log("")
    log("=" * 70)
    log(f"  CITY: {city_name.upper()}")
    log(f"  BG sources   : {config['n_block_groups']:,}")
    log(f"  RAM budget   : {ram_gb:.1f} GB  (shared across all cities)")
    log("=" * 70)

    t_city_start = time.perf_counter()

    # ---------- CSR ----------
    csr, n_nodes = get_csr(city_name, config, force_rebuild=force_rebuild)

    # ---------- Candidates ----------
    cands_file = config["candidates_file"]
    if os.path.exists(cands_file):
        cands        = gpd.read_file(cands_file, engine="pyogrio")
        n_candidates = len(cands)
        del cands
        log(f"\n[Candidates] {n_candidates:,} from {cands_file}")
    else:
        n_candidates = 2450
        log(f"\n[Candidates] WARNING: {cands_file} not found -- "
            f"defaulting to {n_candidates}")

    # Smart direction: run Dijkstra from the SMALLER set
    n_bg   = config["n_block_groups"]
    n_cand = n_candidates
    if n_bg <= n_cand:
        ms_sources = n_bg
        std_runs   = n_cand
        smart_note = "BG sources < candidates -> run from BGs (smart)"
    else:
        ms_sources = n_cand
        std_runs   = n_bg
        smart_note = "candidates < BG sources -> run from candidates (smart)"
    log(f"\n[Direction] {smart_note}")
    log(f"  Multi-source runs from: {ms_sources:,}  |  "
        f"Standard runs from: {std_runs:,}  |  "
        f"Theoretical speedup: {std_runs/ms_sources:.2f}x")

    # ---------- Multi-source Dijkstra (our method) ----------
    ms_results = run_multisource_dijkstra(
        csr, n_nodes, ms_sources, ram_gb, city_name)

    # ---------- Standard Dijkstra (baseline) ----------
    std_results = run_standard_dijkstra(
        csr, n_nodes, std_runs,
        workers=workers, skip=skip_std)

    # ---------- Speedup ----------
    ms_t  = ms_results["multisource_time_s"]
    std_t = std_results["std_time_s"]
    speedup = round(std_t / ms_t, 2) if ms_t > 0 else 0.0

    t_city_total = time.perf_counter() - t_city_start

    log(f"")
    log(f"  *** {city_name} SUMMARY ***")
    log(f"  Multi-source (ours): {_fmt_time(ms_t)}"
        f"  ({'FAIR' if ms_results['multisource_fair'] else 'NON-FAIR'})")
    log(f"  Standard (naive)   : {_fmt_time(std_t)}"
        f"  ({'measured' if not std_results['std_is_extrapolated'] else 'extrapolated'})")
    log(f"  Speedup            : {speedup:.2f}x")
    log(f"  City total time    : {_fmt_time(t_city_total)}")

    result = {
        "city":           city_name,
        "n_road_nodes":   n_nodes,
        "n_road_edges":   int(csr.nnz),
        "n_bg_sources":   n_bg,
        "n_candidates":   n_cand,
        "ram_budget_gb":  round(ram_gb, 1),
        "speedup":        speedup,
        "city_total_s":   round(t_city_total, 1),
        **ms_results,
        **std_results,
    }

    # Save per-city checkpoint immediately (so partial runs aren't lost)
    out_file = os.path.join(OUTPUT_DIR, f"{city_name}_scaling_hpc.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Checkpoint saved: {out_file}")

    return result


# ============================================================
# RESULTS TABLE
# ============================================================
def _fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        return f"{s/60:.1f} min"
    else:
        return f"{s/3600:.2f} hr"


def print_table(all_results):
    log("")
    log("=" * 105)
    log("  PAPER TABLE -- Multi-Source Dijkstra Scaling")
    log("  All cities use the SAME RAM budget -> fair chunk strategy comparison")
    log("=" * 105)

    hdr = (f"  {'City':<15} {'Nodes':>10} {'BG src':>7} {'Cands':>7} "
           f"{'RAM GB':>7} {'Chunks':>7} "
           f"{'Multi-src':>12} {'Std Dijkstra':>14} {'Speedup':>9} {'Fair':>6}")
    log(hdr)
    log(f"  {'-'*98}")

    rows = []
    for r in all_results:
        city     = r["city"]
        nodes    = f"{r['n_road_nodes']:,}"
        bg       = f"{r['n_bg_sources']:,}"
        cands    = f"{r['n_candidates']:,}"
        ram      = f"{r['ram_budget_gb']:.0f}"
        chunks   = str(r.get("multisource_n_chunks", "?"))
        ms_t     = _fmt_time(r["multisource_time_s"])
        std_t    = _fmt_time(r["std_time_s"])
        std_note = "*" if r.get("std_is_extrapolated") else ""
        speedup  = f"{r['speedup']:.2f}x"
        fair     = "YES" if r.get("multisource_fair") else "NO"

        log(f"  {city:<15} {nodes:>10} {bg:>7} {cands:>7} "
            f"{ram:>7} {chunks:>7} "
            f"{ms_t:>12} {std_t+std_note:>14} {speedup:>9} {fair:>6}")

        rows.append({
            "City":                  city,
            "Road Nodes":            r["n_road_nodes"],
            "BG Sources":            r["n_bg_sources"],
            "Candidates":            r["n_candidates"],
            "RAM Budget (GB)":       r["ram_budget_gb"],
            "Multi-src Chunks":      r.get("multisource_n_chunks"),
            "Multi-src Time":        ms_t,
            "Std Dijkstra Time":     std_t + std_note,
            "Speedup":               speedup,
            "Fair (1 chunk)":        "YES" if r.get("multisource_fair") else "NO",
            "Std Extrapolated":      r.get("std_is_extrapolated", False),
        })

    log(f"  {'-'*98}")
    log("  * = extrapolated (--skip-std used)")
    log("")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_file = os.path.join(OUTPUT_DIR, "scaling_table_hpc.csv")
    df.to_csv(csv_file, index=False)
    log(f"  Table saved: {csv_file}")


def load_existing_results(cities):
    """Load previously saved per-city JSON checkpoints."""
    results = []
    for city in cities:
        f = os.path.join(OUTPUT_DIR, f"{city}_scaling_hpc.json")
        if os.path.exists(f):
            with open(f) as fp:
                results.append(json.load(fp))
            log(f"  Loaded: {f}")
        else:
            log(f"  WARNING: no result for {city} ({f} not found)")
    return results


# ============================================================
# DRY RUN  --  show memory plan only, no Dijkstra
# ============================================================
def dry_run(cities_configs, ram_gb):
    log("")
    log("=" * 70)
    log("  DRY RUN -- memory plan only (no Dijkstra executed)")
    log("=" * 70)
    memory_plan(cities_configs, ram_gb)
    log("To run the benchmark:")
    log("  python benchmark_scaling_hpc.py --all --ram-gb <GB>")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="HPC-Ready Multi-Source Dijkstra Scaling Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_scaling_hpc.py --all --ram-gb 80
  python benchmark_scaling_hpc.py --city Charlotte --ram-gb 80
  python benchmark_scaling_hpc.py --all --ram-gb 80 --skip-std
  python benchmark_scaling_hpc.py --all --ram-gb 80 --workers 16
  python benchmark_scaling_hpc.py --dry-run --ram-gb 80
  python benchmark_scaling_hpc.py --table-only
        """
    )
    parser.add_argument("--city",    type=str,
                        help="Single city to run (Charlotte / Atlanta / 'Los Angeles')")
    parser.add_argument("--all",     action="store_true",
                        help="Run all cities")
    parser.add_argument("--ram-gb",  type=float, default=None,
                        help="Total node RAM in GB (SAME budget applied to all "
                             "cities for fair comparison). Auto-detected if omitted.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for standard Dijkstra "
                             "(Linux HPC only, uses fork). Default=1.")
    parser.add_argument("--skip-std", action="store_true",
                        help="Skip full standard Dijkstra; extrapolate from "
                             "first 100 runs instead. Much faster for LA.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print memory plan and exit -- no Dijkstra.")
    parser.add_argument("--table-only", action="store_true",
                        help="Load existing JSON results and reprint table. "
                             "No computation.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild CSR from roads (ignore cache).")
    args = parser.parse_args()

    log("=" * 70)
    log("  MULTI-SOURCE DIJKSTRA SCALING BENCHMARK (HPC Edition)")
    log(f"  Python {sys.version.split()[0]}  |  "
        f"Platform: {sys.platform}")
    log("=" * 70)

    # ---------- table-only mode ----------
    if args.table_only:
        results = load_existing_results(list(CITIES.keys()))
        if results:
            print_table(results)
        else:
            log("No saved results found. Run the benchmark first.")
        return

    # ---------- detect RAM & workers ----------
    ram_gb  = detect_ram_gb(args.ram_gb)
    workers = detect_workers(args.workers)
    if workers > 1:
        log(f"Workers: {workers}  "
            f"({'parallel fork' if sys.platform != 'win32' else 'sequential -- Windows'})")

    # ---------- dry run ----------
    if args.dry_run:
        dry_run(CITIES, ram_gb)
        return

    # ---------- select cities to run ----------
    if args.city:
        if args.city not in CITIES:
            log(f"Unknown city '{args.city}'. Available: {list(CITIES.keys())}")
            sys.exit(1)
        cities_to_run = {args.city: CITIES[args.city]}
    elif args.all:
        cities_to_run = CITIES
    else:
        parser.print_help()
        log("\nSpecify --city <name> or --all")
        sys.exit(1)

    # Print memory plan first so you can abort early if RAM is insufficient
    memory_plan(cities_to_run, ram_gb)

    # ---------- run each city ----------
    all_results = []
    t_total = time.perf_counter()

    for city_name, config in cities_to_run.items():
        try:
            result = run_city(
                city_name, config,
                ram_gb       = ram_gb,
                workers      = workers,
                skip_std     = args.skip_std,
                force_rebuild= args.rebuild,
            )
            all_results.append(result)
        except Exception as exc:
            log(f"ERROR on {city_name}: {exc}")
            import traceback; traceback.print_exc()
            log(f"Skipping {city_name} -- continuing ...")

    # ---------- final table ----------
    if all_results:
        print_table(all_results)

    log(f"\nTotal wall time: {_fmt_time(time.perf_counter() - t_total)}")
    log(f"Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

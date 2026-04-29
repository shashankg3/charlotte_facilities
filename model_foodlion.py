# =========================
# model_foodlion.py  (CHARLOTTE ONLY — FOOD LION TARGET)
#
# Identical to model_approach2.py except:
#   - TARGET brand  : Food Lion  (was Harris Teeter)
#   - COMPETITORS   : Harris Teeter + all other grocery brands
#   - HT ground-truth file is NOT used (Food Lion has no separate GT file;
#     its locations come directly from all_grocery_stores_combined.geojson)
#   - Cache key is bumped so Food Lion cache is stored separately from HT
#
# All scoring logic, road graph, BG distance matrix, and candidate set
# are identical to the HT study — only the store split changes.
#
# CORRECTED VERSION v4 — inherited from model_approach2.py:
#
#   1. FIX: bg_m now computes area_sqmi, pop_per_sqmi, median_income
#      immediately after projection to CRS_M. These fields are required
#      by the /blocks endpoint for block group tooltip display.
#      Without them, Charlotte blocks rendered grey with no data.
#
#   2. CACHE_VERSION bumped to "charlotte_v14_bg_tooltip_fields"
#      so the cache automatically rebuilds on next startup —
#      no need to manually delete cache files or set env vars.
#
# PREVIOUS v3 CHANGES (retained):
#
#   1. DIJKSTRA STRATEGY: Multi-source (Reverse) Dijkstra.
#      Pre-computes distance matrix from all 559 BG sources in ONE
#      scipy call, then looks up candidate distances from that matrix.
#      This matches the paper's recommended algorithm (4.83x speedup).
#
#   2. norm_weight() defined ONCE at module level (no duplicate)
#   3. CIRCUITY_FACTOR = 1.30 named constant
#   4. Access score: A_k = exp(-alpha * t_bar_minutes), alpha=0.05
#   5. Competition index counts ALL stores (HT + competitors)
#   6. Huff model uses K nearest competitors (default K=3)
#
# CACHING:
#   - Parquet caches for GeoDataFrames (fast IO)
#   - joblib cache for road graph arrays + KDTree inputs
#   - CSR adjacency rebuilt from cached edge arrays on load (cheap)
#   - BG distance matrix cached (expensive to compute, cheap to load)
#
# ENV:
#   - MODEL_CACHE_DIR=<path>   (default: cache/)
#   - FORCE_REBUILD_CACHE=1    (rebuild caches)
#   - LOAD_ONLY_CACHE=1        (fail if cache missing; no rebuild)
# =========================

import os
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csr_dijkstra
import networkx as nx
from joblib import dump, load as joblib_load

# ---- DATA FILES ----
STORES_FILE          = "data/all_grocery_stores_combined.geojson"
HT_GROUND_TRUTH_FILE = "data/harris_teeter_ground_truth.geojson"
USE_HT_GROUND_TRUTH  = True   # still load HT GT so HT goes into competitors
HT_DEDUPE_TOL_M      = 50.0

# Food Lion — target brand for this model
TARGET_BRAND = "food lion"

BG_ACS_FILE     = "data/mecklenburg_bg_with_acs.geojson"
BG_POPDENS_FILE = "data/mecklenburg_bg_population_with_density.geojson"
CANDIDATES_FILE = "data/candidates_osm.geojson"  # OSM commercial zones
ROADS_FILE      = "data/charlotte_roads_drive.geojson"
BOUNDARY_FILE   = "data/charlotte_boundary.geojson"

CRS_LL = 4326
CRS_M  = 32119      # NAD83 / North Carolina  (metres)
MILE_M = 1609.344

# Road circuity factor: ratio of road distance to straight-line distance.
CIRCUITY_FACTOR = 1.30

# Negative exponential decay for accessibility index.
# A_k = exp(-ACCESS_ALPHA * t_bar_minutes), alpha=0.05
ACCESS_ALPHA = 0.05

# Census Bureau sentinel value for suppressed median income.
CENSUS_INCOME_SENTINEL = -666_666_666

CACHE_DIR     = "cache"
# BUMPED: forces cache rebuild so new bg_m tooltip fields are saved
CACHE_VERSION = "foodlion_v1_charlotte"
os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================
# UTIL
# =============================================================

def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _file_sig(path: str) -> str:
    if not _exists(path):
        return f"{path}:MISSING"
    st = os.stat(path)
    return f"{path}:{int(st.st_mtime)}:{int(st.st_size)}"


def _cache_key_for_inputs() -> str:
    sig = "|".join([
        CACHE_VERSION,
        _file_sig(STORES_FILE),
        _file_sig(HT_GROUND_TRUTH_FILE) if USE_HT_GROUND_TRUTH else "HT_GT:OFF",
        _file_sig(BG_ACS_FILE),
        _file_sig(BG_POPDENS_FILE),
        _file_sig(CANDIDATES_FILE),
        _file_sig(ROADS_FILE),
        _file_sig(BOUNDARY_FILE),
    ])
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def _cache_base(prefix: str, cache_root: str, cache_key: str) -> str:
    d = os.path.join(cache_root, f"{prefix}_{cache_key}")
    os.makedirs(d, exist_ok=True)
    return d


def _read_to_ll(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        gdf.set_crs(CRS_LL, inplace=True, allow_override=True)
    return gdf.to_crs(CRS_LL)


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


# =============================================================
# NORMALISATION — single definition used everywhere
# =============================================================

def norm_weight(arr: np.ndarray) -> np.ndarray:
    """
    Percentile-robust normalisation to [0.01, 1.0].
    """
    a = np.asarray(arr, float).copy()
    a[~np.isfinite(a)] = np.nan
    valid = a > 0
    if not np.any(valid):
        return np.ones_like(a)
    vals = a[valid]
    lo, hi = np.nanpercentile(vals, [1, 99])
    if hi <= lo:
        w = np.ones_like(a)
    else:
        clipped = np.clip(a, lo, hi)
        w = (clipped - lo) / (hi - lo)
    w[~np.isfinite(w)] = 0.0
    return 0.01 + 0.99 * np.clip(w, 0.0, 1.0)


# =============================================================
# HUFF MODEL
# =============================================================

def huff_share_vs_competitors(
    t_new:   np.ndarray,
    t_comps: np.ndarray,
    beta:    float,
) -> np.ndarray:
    t_new   = np.asarray(t_new,   float)
    t_comps = np.asarray(t_comps, float)
    eps     = 1e-6
    t_new_safe   = np.where(t_new   <= 0, eps,     t_new)
    t_comps_safe = np.where(t_comps <= 0, np.inf,  t_comps)
    A_new  = 1.0 / (t_new_safe ** beta)
    A_comp = np.sum(1.0 / (t_comps_safe ** beta), axis=1)
    return A_new / (A_new + A_comp + 1e-9)


# =============================================================
# ROAD GRAPH HELPERS
# =============================================================

def _build_graph_from_roads(
    roads_m:               gpd.GeoDataFrame,
    snap_tol_m:            float = 1.0,
    simplify_tol_m:        float = 0.0,
    max_vertices_per_line: int   = 5000,
):
    if roads_m is None or roads_m.empty:
        return None, None

    def _snap(x: float, y: float):
        if snap_tol_m and snap_tol_m > 0:
            return (round(x / snap_tol_m) * snap_tol_m,
                    round(y / snap_tol_m) * snap_tol_m)
        return (float(x), float(y))

    G           = nx.Graph()
    node_coords = {}

    for geom in roads_m.geometry:
        if geom is None or geom.is_empty:
            continue
        if simplify_tol_m and simplify_tol_m > 0:
            try:
                geom = geom.simplify(simplify_tol_m, preserve_topology=True)
            except Exception:
                pass

        lines = (list(geom.geoms)
                 if geom.geom_type == "MultiLineString" else [geom])
        for line in lines:
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            if max_vertices_per_line and len(coords) > max_vertices_per_line:
                idx    = np.linspace(0, len(coords) - 1,
                                     max_vertices_per_line).astype(int)
                coords = [coords[i] for i in idx]

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

    if G.number_of_nodes() == 0:
        return None, None

    print(f"[roads] Vertex graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.")
    return G, node_coords


def _build_road_kdtree(node_coords: dict):
    if not node_coords:
        return None, None, None
    node_ids = list(node_coords.keys())
    node_xy  = np.array([node_coords[n] for n in node_ids], dtype=np.float32)
    return cKDTree(node_xy), node_ids, node_xy


def _graph_to_arrays(G: nx.Graph, node_ids: list, node_coords: dict):
    node_xy     = np.array([node_coords[n] for n in node_ids], dtype=np.float32)
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    u_idx, v_idx, w = [], [], []
    for u, v, data in G.edges(data=True):
        u_idx.append(node_to_idx[u])
        v_idx.append(node_to_idx[v])
        w.append(float(data.get("weight", 0.0)))
    return (np.asarray(u_idx, dtype=np.int32),
            np.asarray(v_idx, dtype=np.int32),
            np.asarray(w,     dtype=np.float32),
            node_xy)


def _build_csr_from_edges(
    u_idx:   np.ndarray,
    v_idx:   np.ndarray,
    w:       np.ndarray,
    n_nodes: int,
) -> csr_matrix:
    if n_nodes <= 0:
        return csr_matrix((0, 0), dtype=np.float64)
    if u_idx.size == 0:
        return csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    u2 = np.concatenate([u_idx, v_idx]).astype(np.int32, copy=False)
    v2 = np.concatenate([v_idx, u_idx]).astype(np.int32, copy=False)
    w2 = np.concatenate([w, w]).astype(np.float64, copy=False)
    return csr_matrix((w2, (u2, v2)), shape=(n_nodes, n_nodes), dtype=np.float64)


def _nearest_node_idx(tree: cKDTree, x, y) -> int:
    _, idx = tree.query([x, y], k=1)
    return int(idx)


# =============================================================
# MULTI-SOURCE DIJKSTRA — Pre-compute BG distance matrix
# =============================================================

def precompute_bg_distance_matrix(road_csr, bg_node_ids):
    if road_csr is None or bg_node_ids is None:
        return None

    bg_indices = np.asarray(bg_node_ids, dtype=np.int32)
    n_bg = len(bg_indices)

    import time as _time
    print(f"[dijkstra] Multi-source Dijkstra: {n_bg} BG sources, "
          f"{road_csr.shape[0]} nodes...")

    _t0 = _time.perf_counter()
    bg_dist_matrix = csr_dijkstra(
        road_csr,
        directed=False,
        indices=bg_indices,
        return_predecessors=False,
    )
    _t_ms = (_time.perf_counter() - _t0) * 1000

    print(f"[dijkstra] Distance matrix computed: shape={bg_dist_matrix.shape}, "
          f"time={_t_ms:.0f} ms")
    return bg_dist_matrix


# =============================================================
# STORE DEDUPLICATION
# =============================================================

def _dedupe_append_points_by_distance(
    base_ll: gpd.GeoDataFrame,
    add_ll:  gpd.GeoDataFrame,
    tol_m:   float,
    crs_m:   int = CRS_M,
) -> gpd.GeoDataFrame:
    if add_ll is None or add_ll.empty:
        return base_ll
    if base_ll is None or base_ll.empty:
        return add_ll
    base_m  = base_ll.to_crs(crs_m)
    add_m   = add_ll.to_crs(crs_m)
    base_xy = np.c_[base_m.geometry.x.values, base_m.geometry.y.values]
    tree    = cKDTree(base_xy)
    add_xy  = np.c_[add_m.geometry.x.values, add_m.geometry.y.values]
    d, _    = tree.query(add_xy, k=1)
    keep    = d > float(tol_m)
    if not np.any(keep):
        return base_ll
    return pd.concat([base_ll, add_ll.loc[keep].copy()], ignore_index=True)


# =============================================================
# LOAD & PREP
# =============================================================

def load_all():
    cache_root = os.environ.get("MODEL_CACHE_DIR", CACHE_DIR)
    force      = os.environ.get("FORCE_REBUILD_CACHE", "0").strip().lower() \
                 in ("1", "true", "yes")
    load_only  = os.environ.get("LOAD_ONLY_CACHE",    "0").strip().lower() \
                 in ("1", "true", "yes")

    cache_key = _cache_key_for_inputs()
    base      = _cache_base("foodlion", cache_root, cache_key)

    p_bg       = os.path.join(base, "bg_m.parquet")
    p_cands_m  = os.path.join(base, "cands_m.parquet")
    p_cands_ll = os.path.join(base, "cands_ll.parquet")
    p_ht       = os.path.join(base, "ht_m.parquet")
    p_comp     = os.path.join(base, "comp_m.parquet")
    p_graph    = os.path.join(base, "road_graph.joblib")
    p_bgdist   = os.path.join(base, "bg_dist_matrix.npy")
    expected   = [p_bg, p_cands_m, p_cands_ll, p_ht, p_comp]

    # ---------- FAST PATH ----------
    if (not force) and all(os.path.exists(p) for p in expected):
        try:
            bg_m      = gpd.read_parquet(p_bg)
            cands_m   = gpd.read_parquet(p_cands_m)
            cands_ll  = gpd.read_parquet(p_cands_ll)
            ht_m      = gpd.read_parquet(p_ht)
            comp_m    = gpd.read_parquet(p_comp)

            # ---- Validate cached bg_m has tooltip fields ----
            # If an old cache is missing these columns, force a rebuild
            required_cols = {"area_sqmi", "pop_per_sqmi", "median_income"}
            if not required_cols.issubset(set(bg_m.columns)):
                missing = required_cols - set(bg_m.columns)
                print(f"[cache] Charlotte cache missing bg_m columns {missing}; "
                      f"forcing rebuild.")
                raise ValueError(f"bg_m missing columns: {missing}")

            road_kdtree = road_csr = bg_node_ids = bg_dist_matrix = None

            if os.path.exists(p_graph):
                g           = joblib_load(p_graph)
                node_xy     = g.get("node_xy",     None)
                u_idx       = g.get("u_idx",       None)
                v_idx       = g.get("v_idx",       None)
                w           = g.get("w",           None)
                bg_node_ids = g.get("bg_node_ids", None)

                if (node_xy is not None
                        and isinstance(node_xy, np.ndarray)
                        and node_xy.shape[0] > 0):
                    road_kdtree = cKDTree(node_xy)
                    if u_idx is not None and len(u_idx) > 0:
                        road_csr = _build_csr_from_edges(
                            np.asarray(u_idx, dtype=np.int32),
                            np.asarray(v_idx, dtype=np.int32),
                            np.asarray(w,     dtype=np.float32),
                            int(node_xy.shape[0]),
                        )

            if os.path.exists(p_bgdist) and road_csr is not None:
                bg_dist_matrix = np.load(p_bgdist)
                print(f"[cache] Loaded BG distance matrix: {bg_dist_matrix.shape}")
            elif road_csr is not None and bg_node_ids is not None:
                bg_dist_matrix = precompute_bg_distance_matrix(
                    road_csr, bg_node_ids)
                if bg_dist_matrix is not None:
                    np.save(p_bgdist, bg_dist_matrix)
                    print(f"[cache] Saved BG distance matrix to {p_bgdist}")

            print(f"[cache] Food Lion FAST load from {base}")
            return {
                "stores_ll":       None,
                "ht_m":            ht_m,
                "comp_m":          comp_m,
                "bg_m":            bg_m,
                "cands_ll":        cands_ll,
                "cands_m":         cands_m,
                "road_csr":        road_csr,
                "road_kdtree":     road_kdtree,
                "bg_node_ids":     bg_node_ids,
                "bg_dist_matrix":  bg_dist_matrix,
            }
        except Exception as e:
            print(f"[cache] Charlotte cache load failed; rebuilding. Reason: {e}")

    if load_only:
        raise RuntimeError(
            f"LOAD_ONLY_CACHE=1 but Charlotte cache missing at: {base}")

    # ---------- COLD BUILD ----------
    for f in [STORES_FILE, BG_ACS_FILE]:
        if not _exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    # ---- Stores ----
    stores_ll = _read_to_ll(STORES_FILE)
    if "brand" not in stores_ll.columns:
        raise KeyError("Expected a 'brand' column in STORES_FILE.")

    stores_ll["brand_norm"] = (stores_ll["brand"].astype(str)
                               .str.lower().str.strip())
    # Food Lion is the target; HT ground-truth file loads HT into competitors
    stores_ll["is_ht"] = stores_ll["brand_norm"] == TARGET_BRAND

    if USE_HT_GROUND_TRUTH and _exists(HT_GROUND_TRUTH_FILE):
        ht_gt_ll = _read_to_ll(HT_GROUND_TRUTH_FILE)
        if "brand" not in ht_gt_ll.columns:
            ht_gt_ll["brand"] = "Harris Teeter"
        ht_gt_ll["brand_norm"] = "harris teeter"
        ht_gt_ll["is_ht"]      = False   # HT goes to competitors in FL model
        stores_ll = _dedupe_append_points_by_distance(
            stores_ll, ht_gt_ll, HT_DEDUPE_TOL_M)
        stores_ll["brand_norm"] = (stores_ll["brand"].astype(str)
                                   .str.lower().str.strip())
        stores_ll["is_ht"] = stores_ll["brand_norm"] == TARGET_BRAND

    print(f"[stores] total={len(stores_ll)}  "
          f"FoodLion={int(stores_ll['is_ht'].sum())}  "
          f"comp={int((~stores_ll['is_ht']).sum())}")
    try:
        print("[stores] top brands:",
              stores_ll["brand_norm"].value_counts().head(10).to_dict())
    except Exception:
        pass

    # ---- Block groups ----
    bg_ll = _read_to_ll(BG_ACS_FILE)
    for src, dst in [("med_income", "income"), ("median_income", "income"),
                     ("pop",        "population"), ("pop_total", "population")]:
        if dst not in bg_ll.columns and src in bg_ll.columns:
            bg_ll = bg_ll.rename(columns={src: dst})

    # Sanitise Census income sentinel values
    for inc_col in ["income", "median_income", "med_income"]:
        if inc_col in bg_ll.columns:
            mask = bg_ll[inc_col] <= CENSUS_INCOME_SENTINEL * 0.5
            n_bad = int(mask.sum())
            if n_bad > 0:
                bg_ll.loc[mask, inc_col] = float("nan")
                print(f"[load_all] Replaced {n_bad} Census income sentinels "
                      f"in '{inc_col}' with NaN")

    # Merge density
    try:
        if _exists(BG_POPDENS_FILE):
            bg_pd = gpd.read_file(BG_POPDENS_FILE, engine="pyogrio")
            key   = "GEOID"
            if key in bg_pd.columns and key in bg_ll.columns:
                cols = [key] + [c for c in
                                ["area_sqmi", "pop_per_sqmi", "median_income"]
                                if c in bg_pd.columns]
                bg_ll = bg_ll.merge(
                    bg_pd[cols].drop_duplicates(subset=[key]),
                    on=key, how="left")
                for inc_col in ["median_income"]:
                    if inc_col in bg_ll.columns:
                        mask = bg_ll[inc_col] <= CENSUS_INCOME_SENTINEL * 0.5
                        n_bad = int(mask.sum())
                        if n_bad > 0:
                            bg_ll.loc[mask, inc_col] = float("nan")
                            print(f"[load_all] Replaced {n_bad} Census income "
                                  f"sentinels in density '{inc_col}' with NaN")
                print("[load_all] merged density columns:",
                      [c for c in ["area_sqmi", "pop_per_sqmi", "median_income"]
                       if c in bg_ll.columns])
            else:
                print("[load_all] BG_POPDENS_FILE found but GEOID missing; skipping.")
        else:
            print("[load_all] BG_POPDENS_FILE not found; proceeding without density.")
    except Exception as e:
        print(f"[load_all] Density merge skipped: {e}")

    # ---- Candidates ----
    if _exists(CANDIDATES_FILE):
        cands_ll = _read_to_ll(CANDIDATES_FILE)
        if cands_ll.geometry.is_empty.any():
            c_m = cands_ll.to_crs(CRS_M)
            c_m["geometry"] = c_m.geometry.centroid
            cands_ll = c_m.to_crs(CRS_LL)
    else:
        bg_m_tmp = bg_ll.to_crs(CRS_M).copy()
        bg_m_tmp["geometry"] = bg_m_tmp.geometry.centroid
        cands_ll = bg_m_tmp.to_crs(CRS_LL)

    # ---- Roads ----
    roads_ll = None
    try:
        if (_exists(ROADS_FILE)
                and ROADS_FILE.lower().endswith((".geojson", ".json"))):
            roads_ll = _read_to_ll(ROADS_FILE)
    except Exception as e:
        print(f"[load_all] Roads unreadable: {e}")

    # ---- Boundary clip ----
    try:
        if _exists(BOUNDARY_FILE):
            boundary_ll  = _read_to_ll(BOUNDARY_FILE)[["geometry"]]
            boundary_m   = boundary_ll.to_crs(CRS_M)
            _bufs        = boundary_m.buffer(200)
            boundary_buf = _bufs.geometry.iloc[0] if len(_bufs) == 1 else _bufs.geometry.values[0]

            stores_ll = (stores_ll[stores_ll.to_crs(CRS_M)
                         .within(boundary_buf)].to_crs(CRS_LL))
            cands_ll  = (cands_ll[cands_ll.to_crs(CRS_M)
                         .within(boundary_buf)].to_crs(CRS_LL))

            # Use intersects filter for BGs (consistent across shapely versions).
            # gpd.overlay intersection clips differently in shapely 1.x vs 2.x
            # causing 554 vs 559 BG count discrepancy on HPC vs PC.
            # intersects = include any BG that has ANY overlap with boundary.
            try:
                bg_m_tmp = bg_ll.to_crs(CRS_M)
                bg_mask  = bg_m_tmp.geometry.intersects(boundary_buf)
                bg_ll    = bg_ll[bg_mask.values].reset_index(drop=True)
            except Exception as e:
                print(f"[load_all] BG intersects filter fallback: {e}")
                bg_ll = bg_ll.reset_index(drop=True)

            if roads_ll is not None:
                try:
                    roads_ll = gpd.overlay(roads_ll, boundary_ll,
                                           how="intersection")
                except Exception as e:
                    print(f"[load_all] Roads overlay fallback: {e}")
                    roads_ll = (roads_ll[roads_ll.to_crs(CRS_M)
                                .intersects(boundary_buf)].to_crs(CRS_LL))
    except Exception as e:
        print(f"[load_all] Boundary clip skipped: {e}")

    # ---- Project to metric CRS ----
    ht_m    = stores_ll[stores_ll["is_ht"]].to_crs(CRS_M)[["geometry"]]
    comp_m  = stores_ll[~stores_ll["is_ht"]].to_crs(CRS_M)[["geometry"]]
    bg_m    = bg_ll.to_crs(CRS_M)
    cands_m = cands_ll.to_crs(CRS_M)[["geometry"]]
    roads_m = (roads_ll.to_crs(CRS_M)[["geometry"]]
               if roads_ll is not None else None)

    # ---- FIX: Compute tooltip fields on bg_m ----
    # These are required by the /blocks endpoint for block group display.
    # Nashville had these explicitly; Charlotte was missing them, causing
    # blocks to render grey with zero values in tooltips.
    bg_m["area_sqmi"]    = bg_m.geometry.area / (MILE_M ** 2)
    bg_m["pop_per_sqmi"] = (
        bg_m["population"] / bg_m["area_sqmi"].replace(0, np.nan)
        if "population" in bg_m.columns
        else np.nan
    )
    # Prefer median_income if already merged from BG_POPDENS_FILE,
    # otherwise fall back to income column.
    if "median_income" not in bg_m.columns:
        if "income" in bg_m.columns:
            bg_m["median_income"] = pd.to_numeric(
                bg_m["income"], errors="coerce"
            ).clip(lower=0, upper=300_000)
        else:
            bg_m["median_income"] = np.nan
    else:
        bg_m["median_income"] = pd.to_numeric(
            bg_m["median_income"], errors="coerce"
        ).clip(lower=0, upper=300_000)

    # ---- Road graph ----
    road_csr = road_kdtree = bg_node_ids = graph_payload = None
    bg_dist_matrix = None

    if roads_m is not None and not roads_m.empty:
        road_graph_raw, node_coords = _build_graph_from_roads(roads_m)
        if road_graph_raw is not None and node_coords:
            road_kdtree_raw, road_node_ids, node_xy = \
                _build_road_kdtree(node_coords)

            if "cent" not in bg_m.columns:
                bg_m["cent"] = bg_m.geometry.centroid
            bg_xy_snap  = np.c_[bg_m["cent"].x.values,
                                 bg_m["cent"].y.values]
            _, idxs     = road_kdtree_raw.query(bg_xy_snap, k=1)
            bg_node_ids = idxs.astype(np.int32)

            u_idx, v_idx, w, node_xy = _graph_to_arrays(
                road_graph_raw, road_node_ids, node_coords)

            graph_payload = {
                "u_idx":       u_idx,
                "v_idx":       v_idx,
                "w":           w,
                "node_xy":     node_xy,
                "bg_node_ids": bg_node_ids,
            }

            road_kdtree = cKDTree(node_xy)
            road_csr    = _build_csr_from_edges(
                u_idx, v_idx, w, int(node_xy.shape[0]))

            print(f"[roads] Mapped {len(bg_node_ids)} BGs to road nodes.")
            print(f"[roads] CSR shape: {road_csr.shape}  nnz: {road_csr.nnz}")

            bg_dist_matrix = precompute_bg_distance_matrix(
                road_csr, bg_node_ids)

    print(f"[load_all] stores={len(stores_ll)} | HT={len(ht_m)} | "
          f"comp={len(comp_m)} | BG={len(bg_ll)} | "
          f"cands={len(cands_ll)} | "
          f"roads={0 if roads_m is None else len(roads_m)}")

    # ---- Write cache ----
    try:
        bg_m.to_parquet(p_bg,       index=False)
        cands_m.to_parquet(p_cands_m,  index=False)
        cands_ll.to_parquet(p_cands_ll, index=False)
        ht_m.to_parquet(p_ht,       index=False)
        comp_m.to_parquet(p_comp,    index=False)

        if graph_payload is None:
            graph_payload = {
                "u_idx":       np.zeros((0,),    dtype=np.int32),
                "v_idx":       np.zeros((0,),    dtype=np.int32),
                "w":           np.zeros((0,),    dtype=np.float32),
                "node_xy":     np.zeros((0, 2),  dtype=np.float32),
                "bg_node_ids": None,
            }
        dump(graph_payload, p_graph, compress=3)

        if bg_dist_matrix is not None:
            np.save(p_bgdist, bg_dist_matrix)
            print(f"[cache] Saved BG distance matrix: {bg_dist_matrix.shape}")

        print(f"[cache] Food Lion cache saved to {base}")
    except Exception as e:
        print(f"[cache] Save failed: {e}")

    return {
        "stores_ll":       stores_ll,
        "ht_m":            ht_m,
        "comp_m":          comp_m,
        "bg_m":            bg_m,
        "cands_ll":        cands_ll,
        "cands_m":         cands_m,
        "road_csr":        road_csr,
        "road_kdtree":     road_kdtree,
        "bg_node_ids":     bg_node_ids,
        "bg_dist_matrix":  bg_dist_matrix,
    }


# =============================================================
# DIVERSE SELECTION
# =============================================================

def select_top_diverse(
    out_m:      gpd.GeoDataFrame,
    scores_col: str   = "pair_score",
    N:          int   = 10,
    min_sep_m:  float = 3.0 * MILE_M,
) -> gpd.GeoDataFrame:
    if out_m.empty:
        return out_m

    cand     = out_m.sort_values(scores_col, ascending=False)\
                    .reset_index(drop=True)
    kept_idx = []
    xs       = cand.geometry.x.values
    ys       = cand.geometry.y.values

    for i in range(len(cand)):
        if len(kept_idx) >= N:
            break
        if not kept_idx:
            kept_idx.append(i)
            continue
        dx   = xs[i] - xs[kept_idx]
        dy   = ys[i] - ys[kept_idx]
        dist = np.hypot(dx, dy)
        if np.all(dist >= min_sep_m):
            kept_idx.append(i)

    if not kept_idx:
        kept_idx = [0]
    return cand.iloc[kept_idx].reset_index(drop=True)


# =============================================================
# MAIN SCORING FUNCTION
# =============================================================

def score_all_candidates_like_ht(
    state,
    radius_miles:   float = 5.0,
    beta:           float = 2.0,
    K:              int   = 3,
    max_candidates: int   = None,
    topN:           int   = 10,
    min_sep_miles:  float = 2.5,
    W1:             float = 0.4,
    W2:             float = 0.3,
    W3:             float = 0.3,
    return_all:     bool  = False,
):
    # Enforce weights sum to 1
    wsum = W1 + W2 + W3
    if abs(wsum - 1.0) > 1e-6:
        W1, W2, W3 = W1/wsum, W2/wsum, W3/wsum

    print(f"[score] beta={beta}, K={K}, "
          f"W1={W1:.3f}, W2={W2:.3f}, W3={W3:.3f}, "
          f"radius={radius_miles}mi")

    ht_m           = state["ht_m"]
    comp_m         = state["comp_m"]
    cands_ll       = state["cands_ll"].copy()
    cands_m        = state["cands_m"].copy()
    bg_m           = state["bg_m"]
    road_csr       = state.get("road_csr",        None)
    road_kdtree    = state.get("road_kdtree",     None)
    bg_node_ids    = state.get("bg_node_ids",     None)
    bg_dist_matrix = state.get("bg_dist_matrix",  None)

    rad_m     = radius_miles * MILE_M
    SPEED_MPS = 35 * MILE_M / 3600.0

    if max_candidates is not None and len(cands_ll) > max_candidates:
        cands_ll = cands_ll.iloc[:max_candidates].reset_index(drop=True)
        cands_m  = cands_m.iloc[:max_candidates].reset_index(drop=True)

    cands_ll = cands_ll.reset_index(drop=True)
    cands_m  = cands_m.reset_index(drop=True)

    # ---- BG attributes ----
    bg_cent = bg_m.copy()
    if "cent" not in bg_cent.columns:
        bg_cent["cent"] = bg_cent.geometry.centroid
    bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

    pop_col = _first_existing(bg_cent.columns,
                              ["population", "pop_total", "pop",
                               "POP", "B01001_001E"])
    if pop_col is None:
        raise KeyError("No population column found in bg_m.")
    bg_pop = (pd.to_numeric(bg_cent[pop_col], errors="coerce")
              .fillna(0).to_numpy())

    inc_col = _first_existing(bg_cent.columns,
                              ["income", "median_income",
                               "med_income", "B19013_001E"])
    bg_inc  = (pd.to_numeric(bg_cent[inc_col], errors="coerce").fillna(0)
               .to_numpy() if inc_col else np.zeros_like(bg_pop))

    dens_col = _first_existing(bg_cent.columns,
                               ["pop_per_sqmi", "density", "DENSITY"])
    dens_w   = (norm_weight(pd.to_numeric(bg_cent[dens_col],
                            errors="coerce").fillna(0).to_numpy())
                if dens_col else np.ones_like(bg_pop))

    block_weight = (0.4 * norm_weight(bg_pop)
                  + 0.3 * norm_weight(bg_inc)
                  + 0.3 * dens_w)

    bg_tree = cKDTree(bg_xy)

    # ---- Store trees ----
    if not ht_m.empty or not comp_m.empty:
        xs = np.concatenate([ht_m.geometry.x.values,
                             comp_m.geometry.x.values])
        ys = np.concatenate([ht_m.geometry.y.values,
                             comp_m.geometry.y.values])
        all_stores_xy  = np.c_[xs, ys]
        all_store_tree = cKDTree(all_stores_xy)
    else:
        all_store_tree = None

    if not comp_m.empty:
        comp_xy   = np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values]
        comp_tree = cKDTree(comp_xy)
        k_eff     = min(max(1, K), len(comp_xy))
        dists_bg_to_comp, _ = comp_tree.query(bg_xy, k=k_eff)
        if dists_bg_to_comp.ndim == 1:
            dists_bg_to_comp = dists_bg_to_comp[:, None]
        bg_tcomp = (dists_bg_to_comp / SPEED_MPS) / 60.0
    else:
        bg_tcomp = np.full((len(bg_xy), 1), 60.0)

    # ---- Pre-snap all candidates to road nodes ----
    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    cand_node_ids = None
    if road_kdtree is not None:
        _, cand_node_ids = road_kdtree.query(cand_xy, k=1)
        cand_node_ids = cand_node_ids.astype(np.int32)

    # ---- Per-candidate scoring loop ----
    metr_s10k      = []
    metr_inc       = []
    metr_access    = []
    metr_potential = []

    use_precomputed = (bg_dist_matrix is not None
                       and cand_node_ids is not None
                       and bg_node_ids is not None)

    if use_precomputed:
        print(f"[score] Using pre-computed BG distance matrix "
              f"(Multi-source Dijkstra, {bg_dist_matrix.shape[0]} sources)")
    else:
        print(f"[score] WARNING: BG distance matrix not available, "
              f"falling back to per-candidate Dijkstra")

    for i, (cx, cy) in enumerate(cand_xy):
        idxs = bg_tree.query_ball_point([cx, cy], r=rad_m)
        if idxs:
            idxs_arr = np.asarray(idxs, dtype=np.int32)
            pop_buf  = float(np.sum(bg_pop[idxs_arr]))
            if pop_buf > 0:
                w_pop    = bg_pop[idxs_arr]
                inc_vals = bg_inc[idxs_arr]
                mask     = (w_pop > 0) & np.isfinite(inc_vals) & (inc_vals > 0)
                incM     = (float(np.average(inc_vals[mask], weights=w_pop[mask]))
                            if np.any(mask) else 0.0)
                incM     = float(np.clip(incM, 0, 300_000))
            else:
                incM = 0.0
        else:
            idxs_arr = None
            pop_buf  = 0.0
            incM     = 0.0

        if all_store_tree is not None and pop_buf > 0:
            store_ct = len(all_store_tree.query_ball_point([cx, cy], r=rad_m))
            s10k     = store_ct / (pop_buf / 10_000.0)
        else:
            s10k = 0.0

        metr_s10k.append(s10k)
        metr_inc.append(incM)

        if idxs_arr is not None and len(idxs_arr) > 0:
            d_new  = np.hypot(bg_xy[idxs_arr, 0] - cx,
                              bg_xy[idxs_arr, 1] - cy)
            t_new  = (d_new / SPEED_MPS) / 60.0
            share  = huff_share_vs_competitors(
                t_new, bg_tcomp[idxs_arr, :], beta)
            metr_potential.append(
                float(np.nansum(block_weight[idxs_arr] * share)))
        else:
            metr_potential.append(0.0)

        access_score = 0.0
        if idxs_arr is not None and use_precomputed:
            cand_node   = int(cand_node_ids[i])
            dist_m_road = bg_dist_matrix[:, cand_node][idxs_arr]
            t_road_min  = (dist_m_road / SPEED_MPS) / 60.0
            w_loc       = block_weight[idxs_arr]
            valid       = np.isfinite(t_road_min) & (t_road_min > 0) & (w_loc > 0)
            if valid.any():
                t_bar_min    = float(
                    np.nansum(w_loc[valid] * t_road_min[valid])
                    / np.nansum(w_loc[valid]))
                access_score = float(
                    np.clip(np.exp(-ACCESS_ALPHA * t_bar_min), 0.01, 1.0))
        elif (idxs_arr is not None
                and road_csr is not None
                and road_kdtree is not None):
            cand_node = _nearest_node_idx(road_kdtree, cx, cy)
            cutoff_m  = rad_m * CIRCUITY_FACTOR
            dist_all  = csr_dijkstra(
                road_csr,
                directed=False,
                indices=int(cand_node),
                return_predecessors=False,
                limit=float(cutoff_m),
            )
            local_nodes = bg_node_ids[idxs_arr].astype(np.int32, copy=False)
            dist_m_road = dist_all[local_nodes].astype(np.float64, copy=False)
            t_road_min  = (dist_m_road / SPEED_MPS) / 60.0
            w_loc       = block_weight[idxs_arr]
            valid       = np.isfinite(t_road_min) & (t_road_min > 0) & (w_loc > 0)
            if valid.any():
                t_bar_min    = float(
                    np.nansum(w_loc[valid] * t_road_min[valid])
                    / np.nansum(w_loc[valid]))
                access_score = float(
                    np.clip(np.exp(-ACCESS_ALPHA * t_bar_min), 0.01, 1.0))

        metr_access.append(access_score)

    # ---- Aggregate ----
    s10k_arr      = np.asarray(metr_s10k,      float)
    inc_arr       = np.asarray(metr_inc,       float)
    access_arr    = np.asarray(metr_access,    float)
    potential_arr = np.asarray(metr_potential, float)

    potential_norm = norm_weight(potential_arr)
    access_norm    = norm_weight(access_arr)
    s10k_norm      = norm_weight(s10k_arr)

    final_scores = W1 * potential_norm + W2 * access_norm - W3 * s10k_norm

    ps    = np.asarray(final_scores, float)
    inten = (ps - np.nanmin(ps)) / (np.nanmax(ps) - np.nanmin(ps) + 1e-9)

    out_ll = cands_ll.copy()
    out_ll["pair_score"]      = np.round(ps,             4)
    out_ll["intensity"]       = np.round(inten,          4)
    out_ll["potential_norm"]  = np.round(potential_norm, 4)
    out_ll["access_norm"]     = np.round(access_norm,    4)
    out_ll["s10k_norm"]       = np.round(s10k_norm,      4)
    out_ll["potential_raw"]   = np.round(potential_arr,  6)
    out_ll["access_score_dj"] = np.round(access_arr,     4)
    out_ll["stores_per_10k"]  = np.round(s10k_arr,       4)
    out_ll["income_med"]      = np.round(inc_arr,        2)

    ht_gdf      = None
    heat_points = []

    out_m       = out_ll.to_crs(CRS_M)
    diverse_top = select_top_diverse(
        out_m[["geometry", "pair_score", "stores_per_10k",
               "income_med", "access_score_dj"]].copy(),
        scores_col="pair_score",
        N=topN,
        min_sep_m=float(min_sep_miles) * MILE_M,
    )
    top_n = diverse_top.to_crs(CRS_LL).reset_index(drop=True)

    if return_all:
        return top_n, heat_points, ht_gdf, out_ll

    return top_n, heat_points, ht_gdf


# =============================================================
# LOCAL SCORING CONTEXT
# =============================================================

def _prep_scoring_context(state, beta=2.0, K=3):
    ht_m           = state["ht_m"]
    comp_m         = state["comp_m"]
    bg_m           = state["bg_m"]
    road_csr       = state.get("road_csr",        None)
    road_kdtree    = state.get("road_kdtree",     None)
    bg_node_ids    = state.get("bg_node_ids",     None)
    bg_dist_matrix = state.get("bg_dist_matrix",  None)
    SPEED_MPS      = 35 * MILE_M / 3600.0

    bg_cent = bg_m.copy()
    if "cent" not in bg_cent.columns:
        bg_cent["cent"] = bg_cent.geometry.centroid
    bg_xy   = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]
    bg_tree = cKDTree(bg_xy)

    pop_col = _first_existing(bg_cent.columns,
                              ["population", "pop_total", "pop",
                               "POP", "B01001_001E"])
    if pop_col is None:
        raise KeyError("No population column found in bg_m.")
    bg_pop  = (pd.to_numeric(bg_cent[pop_col], errors="coerce")
               .fillna(0).to_numpy())

    inc_col = _first_existing(bg_cent.columns,
                              ["income", "median_income",
                               "med_income", "B19013_001E"])
    bg_inc  = (pd.to_numeric(bg_cent[inc_col], errors="coerce").fillna(0)
               .to_numpy() if inc_col else np.zeros_like(bg_pop))

    dens_col = _first_existing(bg_cent.columns,
                               ["pop_per_sqmi", "density", "DENSITY"])
    dens_w   = (norm_weight(pd.to_numeric(bg_cent[dens_col],
                            errors="coerce").fillna(0).to_numpy())
                if dens_col else np.ones_like(bg_pop))

    block_weight = (0.4 * norm_weight(bg_pop)
                  + 0.3 * norm_weight(bg_inc)
                  + 0.3 * dens_w)

    if not comp_m.empty:
        comp_xy   = np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values]
        comp_tree = cKDTree(comp_xy)
        k_eff     = min(max(1, int(K)), len(comp_xy))
        d, _      = comp_tree.query(bg_xy, k=k_eff)
        if d.ndim == 1:
            d = d[:, None]
        bg_tcomp  = (d / SPEED_MPS) / 60.0
    else:
        bg_tcomp = np.full((len(bg_xy), 1), 60.0)

    if not ht_m.empty or not comp_m.empty:
        xs = np.concatenate([ht_m.geometry.x.values,
                             comp_m.geometry.x.values])
        ys = np.concatenate([ht_m.geometry.y.values,
                             comp_m.geometry.y.values])
        all_store_tree = cKDTree(np.c_[xs, ys])
    else:
        all_store_tree = None

    return {
        "bg_xy":           bg_xy,
        "bg_tree":         bg_tree,
        "bg_pop":          bg_pop,
        "bg_inc":          bg_inc,
        "block_weight":    block_weight,
        "bg_tcomp":        bg_tcomp,
        "all_store_tree":  all_store_tree,
        "road_csr":        road_csr,
        "road_kdtree":     road_kdtree,
        "bg_node_ids":     bg_node_ids,
        "bg_dist_matrix":  bg_dist_matrix,
        "SPEED_MPS":       SPEED_MPS,
        "beta":            float(beta),
    }


def _score_xy_points(
    ctx,
    cand_xy,
    radius_miles=2.0,
    W1=0.4, W2=0.3, W3=0.3,
):
    cand_xy   = np.asarray(cand_xy, float)
    rad_m     = float(radius_miles) * MILE_M

    bg_xy          = ctx["bg_xy"]
    bg_tree        = ctx["bg_tree"]
    bg_pop         = ctx["bg_pop"]
    bg_inc         = ctx["bg_inc"]
    block_weight   = ctx["block_weight"]
    bg_tcomp       = ctx["bg_tcomp"]
    all_store_tree = ctx["all_store_tree"]
    road_csr       = ctx["road_csr"]
    road_kdtree    = ctx["road_kdtree"]
    bg_node_ids    = ctx["bg_node_ids"]
    bg_dist_matrix = ctx["bg_dist_matrix"]
    SPEED_MPS      = ctx["SPEED_MPS"]
    beta           = ctx["beta"]

    use_precomputed = (bg_dist_matrix is not None and road_kdtree is not None)

    cand_node_ids = None
    if road_kdtree is not None:
        _, cand_node_ids = road_kdtree.query(cand_xy, k=1)
        cand_node_ids = cand_node_ids.astype(np.int32)

    pops, incs, s10ks, access_scores, potentials = [], [], [], [], []

    for i, (cx, cy) in enumerate(cand_xy):
        idxs = bg_tree.query_ball_point([cx, cy], r=rad_m)
        idxs_arr = np.asarray(idxs, dtype=np.int32) if idxs else None

        if idxs_arr is not None:
            pop_buf = float(np.sum(bg_pop[idxs_arr]))
            if pop_buf > 0:
                w_pop    = bg_pop[idxs_arr]
                inc_vals = bg_inc[idxs_arr]
                mask     = (w_pop > 0) & np.isfinite(inc_vals) & (inc_vals > 0)
                incM     = (float(np.average(inc_vals[mask], weights=w_pop[mask]))
                            if np.any(mask) else 0.0)
                incM     = float(np.clip(incM, 0, 300_000))
            else:
                incM = 0.0
        else:
            pop_buf = 0.0
            incM    = 0.0

        if all_store_tree is not None and pop_buf > 0:
            store_ct = len(all_store_tree.query_ball_point([cx, cy], r=rad_m))
            s10k     = store_ct / (pop_buf / 10_000.0)
        else:
            s10k = 0.0

        if idxs_arr is not None and len(idxs_arr) > 0:
            d_new  = np.hypot(bg_xy[idxs_arr, 0] - cx,
                              bg_xy[idxs_arr, 1] - cy)
            t_new  = (d_new / SPEED_MPS) / 60.0
            share  = huff_share_vs_competitors(
                t_new, bg_tcomp[idxs_arr, :], beta)
            potential = float(np.nansum(block_weight[idxs_arr] * share))
        else:
            potential = 0.0

        access_score = 0.0
        if idxs_arr is not None and use_precomputed:
            cand_node   = int(cand_node_ids[i])
            dist_m_road = bg_dist_matrix[:, cand_node][idxs_arr]
            t_road_min  = (dist_m_road / SPEED_MPS) / 60.0
            w_loc       = block_weight[idxs_arr]
            valid       = np.isfinite(t_road_min) & (t_road_min > 0) & (w_loc > 0)
            if valid.any():
                t_bar_min    = float(
                    np.nansum(w_loc[valid] * t_road_min[valid])
                    / np.nansum(w_loc[valid]))
                access_score = float(
                    np.clip(np.exp(-ACCESS_ALPHA * t_bar_min), 0.01, 1.0))
        elif (idxs_arr is not None
                and road_csr is not None
                and road_kdtree is not None
                and bg_node_ids is not None):
            cand_node   = _nearest_node_idx(road_kdtree, cx, cy)
            cutoff_m    = rad_m * CIRCUITY_FACTOR
            dist_all    = csr_dijkstra(
                road_csr, directed=False,
                indices=int(cand_node),
                return_predecessors=False,
                limit=float(cutoff_m))
            local_nodes = bg_node_ids[idxs_arr].astype(np.int32, copy=False)
            dist_m_road = dist_all[local_nodes].astype(np.float64, copy=False)
            t_road_min  = (dist_m_road / SPEED_MPS) / 60.0
            w_loc       = block_weight[idxs_arr]
            valid       = np.isfinite(t_road_min) & (t_road_min > 0) & (w_loc > 0)
            if valid.any():
                t_bar_min    = float(
                    np.nansum(w_loc[valid] * t_road_min[valid])
                    / np.nansum(w_loc[valid]))
                access_score = float(
                    np.clip(np.exp(-ACCESS_ALPHA * t_bar_min), 0.01, 1.0))

        pops.append(pop_buf)
        incs.append(incM)
        s10ks.append(s10k)
        access_scores.append(access_score)
        potentials.append(potential)

    pops          = np.asarray(pops,          float)
    incs          = np.asarray(incs,          float)
    s10ks         = np.asarray(s10ks,         float)
    access_scores = np.asarray(access_scores, float)
    potentials    = np.asarray(potentials,    float)

    pot_norm    = norm_weight(potentials)
    access_norm = norm_weight(access_scores)
    s10k_norm   = norm_weight(s10ks)

    score = W1 * pot_norm + W2 * access_norm - W3 * s10k_norm

    return score, pops, incs, s10ks, access_scores


def get_local_validation_points_payload(
    state,
    ht_index=0,
    local_radius_miles=2.0,
    score_radius_miles=None,
    beta=2.0,
    K=3,
    W1=0.4, W2=0.3, W3=0.3,
):
    if score_radius_miles is None:
        score_radius_miles = local_radius_miles

    ht_m    = state["ht_m"].reset_index(drop=True)
    cands_m = state["cands_m"]

    if ht_m.empty:
        raise ValueError("No HT stores in state['ht_m'].")
    if not (0 <= ht_index < len(ht_m)):
        raise IndexError(
            f"ht_index must be 0–{len(ht_m)-1}, got {ht_index}.")

    ctx         = _prep_scoring_context(state, beta=beta, K=K)
    cand_xy_all = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    cand_tree   = cKDTree(cand_xy_all)

    hx, hy        = (float(ht_m.loc[ht_index].geometry.x),
                     float(ht_m.loc[ht_index].geometry.y))
    rad_local_m   = float(local_radius_miles) * MILE_M
    idxs          = cand_tree.query_ball_point([hx, hy], r=rad_local_m)
    local_xy      = cand_xy_all[idxs] if idxs else np.empty((0, 2))
    all_xy        = np.vstack([[hx, hy], local_xy])

    score, pop_buf, inc_med, s10k, access = _score_xy_points(
        ctx, all_xy,
        radius_miles=float(score_radius_miles),
        W1=W1, W2=W2, W3=W3,
    )

    pts_ll    = (gpd.GeoSeries(
                    gpd.points_from_xy(all_xy[:, 0], all_xy[:, 1]),
                    crs=CRS_M)
                 .to_crs(CRS_LL))
    ht_score  = float(score[0])
    ht_pct    = float(np.mean(score <= ht_score) * 100.0)
    ht_rank   = int(np.argsort(-score).tolist().index(0) + 1)

    points = []
    for i, geom in enumerate(pts_ll):
        points.append({
            "lat":             float(geom.y),
            "lon":             float(geom.x),
            "is_existing_ht":  bool(i == 0),
            "score":           float(score[i]),
            "population_buf":  float(pop_buf[i]),
            "income_med":      float(inc_med[i]),
            "stores_per_10k":  float(s10k[i]),
            "access_score_dj": float(access[i]),
        })

    return {
        "ht_index":            int(ht_index),
        "local_radius_miles":  float(local_radius_miles),
        "score_radius_miles":  float(score_radius_miles),
        "ht_percentile_local": ht_pct,
        "ht_rank_local":       ht_rank,
        "n_points":            int(len(points)),
        "points":              points,
    }
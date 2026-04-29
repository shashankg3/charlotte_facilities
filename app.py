"""
app.py  —  Charlotte Facilities Site-Selection Tool
-----------------------------------------------------
Unified app for Harris Teeter + Food Lion.
Select brand from the dropdown; model and defaults switch automatically.

Run:
    C:\\Python312\\python.exe app.py
Then open:  http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import logging, os, json, math

import numpy as np
from scipy.spatial import cKDTree
import geopandas as gpd

def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _parse_float(raw, default):
    try: return float(raw)
    except Exception: return default

def _parse_int(raw, default):
    try: return int(raw)
    except Exception: return default

# ── Import both models ────────────────────────────────────────
from model_ht        import load_all as load_ht,  score_all_candidates_like_ht as score_ht
from model_foodlion  import load_all as load_fl,  score_all_candidates_like_ht as score_fl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

STATE_HT = None
STATE_FL = None

# ── Optimal params (loaded from Results_final) ────────────────
def _load_params(path, defaults):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return defaults

HT_PARAMS = _load_params("Results_final/HT_results/V5/optimal_params.json", {
    "radius_miles": 4.0, "beta": 5.0, "K": 8,
    "W1": 0.75, "W2": 0.05, "W3": 0.20
})
FL_PARAMS = _load_params("Results_final/FL_results/V5/optimal_params.json", {
    "radius_miles": 5.5, "beta": 3.5, "K": 6,
    "W1": 0.10, "W2": 0.05, "W3": 0.85
})

BRAND_PARAMS = {
    "ht": HT_PARAMS,
    "fl": FL_PARAMS,
}

def _should_init_now():
    if not app.debug: return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"

def load_models():
    global STATE_HT, STATE_FL
    logger.info("Loading Harris Teeter model...")
    STATE_HT = load_ht()
    logger.info(f"  HT: {len(STATE_HT['ht_m'])} stores, {len(STATE_HT['comp_m'])} competitors")
    logger.info("Loading Food Lion model...")
    STATE_FL = load_fl()
    logger.info(f"  FL: {len(STATE_FL['ht_m'])} stores, {len(STATE_FL['comp_m'])} competitors")

def _state(brand):
    global STATE_HT, STATE_FL
    if brand == "fl":
        if STATE_FL is None: STATE_FL = load_fl()
        return STATE_FL, score_fl
    else:
        if STATE_HT is None: STATE_HT = load_ht()
        return STATE_HT, score_ht

if _should_init_now():
    load_models()

# ── Helpers ───────────────────────────────────────────────────
def gdf_points_to_list(gdf_m):
    if gdf_m is None or gdf_m.empty: return []
    ll = gdf_m.to_crs(4326)
    return [[float(g.y), float(g.x)] for g in ll.geometry]

def _comp_with_brands(state):
    comp_ll = state["comp_m"].to_crs(4326).reset_index(drop=True)
    raw = gpd.read_file("data/all_grocery_stores_combined.geojson").to_crs(4326)
    if "brand" not in raw.columns:
        raw["brand"] = "Competitor"
    comp_xy = np.c_[comp_ll.geometry.x.values, comp_ll.geometry.y.values]
    raw_xy  = np.c_[raw.geometry.x.values,     raw.geometry.y.values]
    _, idxs = cKDTree(raw_xy).query(comp_xy, k=1)
    result = []
    for i, row in comp_ll.iterrows():
        b = str(raw.iloc[idxs[i]]["brand"]).strip().title()
        if b.lower() in ("nan","none",""): b = "Competitor"
        result.append({"lat": float(row.geometry.y),
                        "lon": float(row.geometry.x),
                        "brand": b})
    return result


# =============================================================
# ROUTES
# =============================================================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "ht_loaded": STATE_HT is not None,
        "fl_loaded": STATE_FL is not None,
    })


@app.route("/")
def home():
    return render_template("index.html",
                           ht_params=json.dumps(HT_PARAMS),
                           fl_params=json.dumps(FL_PARAMS))


@app.get("/params")
def params():
    brand = request.args.get("brand", "ht").lower()
    return jsonify(BRAND_PARAMS.get(brand, HT_PARAMS))


@app.route("/recompute", methods=["GET", "POST"])
def recompute():
    raw = (request.get_json(silent=True) or {}) if request.method == "POST" \
          else request.args

    brand  = str(raw.get("brand", "ht")).lower()
    state, score_fn = _state(brand)
    p      = BRAND_PARAMS.get(brand, HT_PARAMS)

    radius = _parse_float(raw.get("radius_miles", p["radius_miles"]), p["radius_miles"])
    beta   = _parse_float(raw.get("beta",         p["beta"]),         p["beta"])
    K      = _parse_int(  raw.get("K",            p["K"]),            p["K"])
    W1     = _parse_float(raw.get("W1",           p["W1"]),           p["W1"])
    W2     = _parse_float(raw.get("W2",           p["W2"]),           p["W2"])
    W3     = _parse_float(raw.get("W3",           p["W3"]),           p["W3"])
    topN   = _parse_int(  raw.get("topN",         10),                10)

    logger.info(f"[recompute] brand={brand} W1={W1} W2={W2} W3={W3} "
                f"beta={beta} radius={radius} K={K}")

    try:
        top_n, heat_points, _ = score_fn(
            state, radius_miles=radius, beta=beta, K=K,
            topN=topN, W1=W1, W2=W2, W3=W3,
        )
    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500

    top_payload = []
    for rank, (_, r) in enumerate(top_n.iterrows(), start=1):
        g = r.geometry
        top_payload.append({
            "rank":            rank,
            "lat":             _safe_float(g.y),
            "lon":             _safe_float(g.x),
            "score":           _safe_float(r.get("pair_score",      0.0)),
            "stores_per_10k":  _safe_float(r.get("stores_per_10k",  0.0)),
            "income_med":      _safe_float(r.get("income_med",       0.0)),
            "access_score_dj": _safe_float(r.get("access_score_dj", 0.0)),
        })

    brand_label = "Harris Teeter" if brand == "ht" else "Food Lion"

    return jsonify({
        "ok":          True,
        "brand":       brand_label,
        "heat_points": heat_points,
        "top10":       top_payload,
        "existing_ht": gdf_points_to_list(state["ht_m"]),
        "competitors": _comp_with_brands(state),
        "n_stores":    len(state["ht_m"]),
        "n_comp":      len(state["comp_m"]),
        "n_cands":     len(state["cands_m"]),
    })


@app.get("/blocks")
def blocks():
    cache_path = os.path.join("cache", "blocks_charlotte.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return app.response_class(f.read(), mimetype="application/json")

    raw_file = "data/mecklenburg_bg_population_with_density.geojson"
    if os.path.exists(raw_file):
        bg = gpd.read_file(raw_file, engine="pyogrio").to_crs(4326)
    else:
        bg = STATE_HT["bg_m"].to_crs(4326)

    cols = set(bg.columns)
    features = []
    for _, row in bg.iterrows():
        if   "median_income" in cols: med_inc = _safe_float(row["median_income"])
        elif "income"        in cols: med_inc = _safe_float(row["income"])
        else:                         med_inc = 0.0
        features.append({
            "type":     "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "GEOID":         row["GEOID"]         if "GEOID"         in cols else None,
                "population":    _safe_float(row["population"]   if "population"   in cols else 0),
                "area_sqmi":     _safe_float(row["area_sqmi"]    if "area_sqmi"    in cols else 0),
                "pop_per_sqmi":  _safe_float(row["pop_per_sqmi"] if "pop_per_sqmi" in cols else 0),
                "median_income": med_inc,
            },
        })

    result = {"type": "FeatureCollection", "features": features}
    os.makedirs("cache", exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(result, f, allow_nan=False)
    except Exception as e:
        logger.warning(f"[blocks] cache write failed: {e}")

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Charlotte Facilities app on http://localhost:{port}")
    app.run(debug=True, use_reloader=False, port=port)

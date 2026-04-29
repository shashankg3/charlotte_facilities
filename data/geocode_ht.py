"""
Geocode Harris Teeter CSV -> GeoJSON (EPSG:4326)

Input CSV columns expected:
store_id,brand,store_name,address,city,state,zip,role

Output:
- harris_teeter_ground_truth.geojson
- harris_teeter_geocode_cache.csv (so you don't re-hit the API)
"""

import time
import json
import pathlib
import urllib.parse
from typing import Optional, Dict, Any

import pandas as pd
import requests


# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "harris_teeter_ground_truth.csv"
OUT_GEOJSON = "harris_teeter_ground_truth.geojson"
CACHE_CSV = "harris_teeter_geocode_cache.csv"

# Nominatim usage policy expects a real User-Agent with contact info.
USER_AGENT = "UNCC-FacilityLocationResearch/1.0 (contact: your_email@uncc.edu)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Rate limiting: be polite (Nominatim recommends 1 req/sec or slower).
SLEEP_SECONDS = 1.1

# Retries for transient errors
MAX_RETRIES = 3


# ----------------------------
# Helpers
# ----------------------------
def build_query(row: pd.Series) -> str:
    parts = [
        str(row.get("address", "")),
        str(row.get("city", "")),
        str(row.get("state", "")),
        str(row.get("zip", "")),
        "USA",
    ]
    return ", ".join([p.strip() for p in parts if p and p.strip()])


def nominatim_geocode(q: str) -> Optional[Dict[str, Any]]:
    """
    Returns dict with lat/lon + some metadata, or None if not found.
    """
    params = {
        "format": "json",
        "q": q,
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                # Too many requests; back off.
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            return data[0]
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2 * attempt)

    return None


def load_or_init_cache(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["store_id", "query", "lat", "lon", "display_name", "raw_json"])


def save_geojson(df: pd.DataFrame, out_path: str) -> None:
    features = []
    for _, row in df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            # Skip ungeocoded rows; alternatively include them with null geometry.
            continue

        props = {
            "store_id": row.get("store_id"),
            "brand": row.get("brand"),
            "store_name": row.get("store_name"),
            "address": row.get("address"),
            "city": row.get("city"),
            "state": row.get("state"),
            "zip": str(row.get("zip")),
            "role": row.get("role"),
            "geocode_display_name": row.get("display_name"),
        }

        feat = {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                # GeoJSON is lon,lat (x,y)
                "coordinates": [float(row["lon"]), float(row["lat"])],
            },
        }
        features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)


# ----------------------------
# Main
# ----------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    required = ["store_id", "brand", "store_name", "address", "city", "state", "zip", "role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    cache = load_or_init_cache(CACHE_CSV)
    cache_idx = {str(r["store_id"]): r for _, r in cache.iterrows()}

    out_rows = []
    newly_geocoded = 0

    for _, row in df.iterrows():
        store_id = str(row["store_id"])
        q = build_query(row)

        # Cache hit?
        if store_id in cache_idx and pd.notna(cache_idx[store_id]["lat"]) and pd.notna(cache_idx[store_id]["lon"]):
            lat = cache_idx[store_id]["lat"]
            lon = cache_idx[store_id]["lon"]
            display_name = cache_idx[store_id].get("display_name", "")
            out_rows.append({**row.to_dict(), "query": q, "lat": lat, "lon": lon, "display_name": display_name})
            continue

        # Geocode
        print(f"Geocoding {store_id}: {q}")
        result = nominatim_geocode(q)
        time.sleep(SLEEP_SECONDS)

        if result is None:
            print(f"  -> NOT FOUND")
            out_rows.append({**row.to_dict(), "query": q, "lat": None, "lon": None, "display_name": ""})
            # update cache
            cache = pd.concat([cache, pd.DataFrame([{
                "store_id": store_id,
                "query": q,
                "lat": None,
                "lon": None,
                "display_name": "",
                "raw_json": "",
            }])], ignore_index=True)
            continue

        lat = result.get("lat")
        lon = result.get("lon")
        display_name = result.get("display_name", "")
        newly_geocoded += 1
        print(f"  -> {lat}, {lon}")

        out_rows.append({**row.to_dict(), "query": q, "lat": lat, "lon": lon, "display_name": display_name})

        cache = pd.concat([cache, pd.DataFrame([{
            "store_id": store_id,
            "query": q,
            "lat": lat,
            "lon": lon,
            "display_name": display_name,
            "raw_json": json.dumps(result),
        }])], ignore_index=True)

    # Save cache + geojson
    cache.to_csv(CACHE_CSV, index=False)
    out_df = pd.DataFrame(out_rows)

    # Basic QA: show misses
    misses = out_df[out_df["lat"].isna() | out_df["lon"].isna()]
    if len(misses) > 0:
        print("\nWARNING: Some addresses were not geocoded:")
        print(misses[["store_id", "address", "city", "state", "zip", "query"]].to_string(index=False))

    save_geojson(out_df, OUT_GEOJSON)
    print(f"\nDone. Wrote {OUT_GEOJSON} (newly geocoded: {newly_geocoded}, total: {len(out_df)})")


if __name__ == "__main__":
    main()

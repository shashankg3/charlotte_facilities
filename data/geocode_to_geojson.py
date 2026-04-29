import time
import json
import pandas as pd
import requests

INPUT_CSV = "all_grocery_stores_combined.csv"
OUT_GEOJSON = "all_grocery_stores_combined.geojson"
OUT_CSV = "all_grocery_stores_combined_geocoded.csv"

# IMPORTANT: Put your real email here per Nominatim policy
USER_AGENT = "UNCC-FacilityLocation/1.0 (contact: your_email@uncc.edu)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
SLEEP_SECONDS = 1.1
MAX_RETRIES = 3


def clean_zip(z):
    """Convert zip to a clean string (fixes 28226.0 problem)."""
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return ""
    s = str(z).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def geocode_one(query: str):
    params = {
        "format": "json",
        "q": query,
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
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


def main():
    # Force zip to be read as string to avoid float coercion
    df = pd.read_csv(INPUT_CSV, dtype={"zip": str})

    # expected columns (from our combined CSV)
    # store_id, brand, store_name, address, city, state, zip, role
    for col in ["store_id", "brand", "address", "city", "state"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    lats, lons, postcodes, display_names = [], [], [], []

    for _, row in df.iterrows():
        zip_code = clean_zip(row.get("zip", ""))

        q = f'{row["address"]}, {row["city"]}, {row["state"]} {zip_code}, USA'
        q = " ".join(q.split())  # normalize whitespace

        print(f'Geocoding {row["store_id"]} ({row["brand"]}): {q}')
        res = geocode_one(q)
        time.sleep(SLEEP_SECONDS)

        if res is None:
            lats.append(None)
            lons.append(None)
            postcodes.append(zip_code)
            display_names.append("")
            print("  -> NOT FOUND")
            continue

        lat = res.get("lat")
        lon = res.get("lon")
        display = res.get("display_name", "")
        addr = res.get("address", {}) or {}
        postcode = addr.get("postcode", "") or zip_code

        lats.append(lat)
        lons.append(lon)
        postcodes.append(postcode)
        display_names.append(display)

        print(f"  -> {lat}, {lon} ({postcode})")

    df["latitude"] = lats
    df["longitude"] = lons
    df["zip"] = postcodes
    df["geocode_display_name"] = display_names

    # Save geocoded CSV too (useful for debugging)
    df.to_csv(OUT_CSV, index=False)

    # Build GeoJSON
    features = []
    for _, row in df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        props = {
            "store_id": row["store_id"],
            "brand": row["brand"],
            "name": row.get("store_name", ""),
            "address": row.get("address", ""),
            "city": row.get("city", ""),
            "state": row.get("state", ""),
            "zip": clean_zip(row.get("zip", "")),
            "role": row.get("role", ""),
            "geocode_display_name": row.get("geocode_display_name", ""),
        }

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],  # lon,lat
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    misses = df[df["latitude"].isna() | df["longitude"].isna()]
    print(f"\nDone. Wrote: {OUT_GEOJSON}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Geocoded: {len(df) - len(misses)} / {len(df)}")
    if len(misses) > 0:
        print("\nNOT FOUND rows (fix these addresses):")
        print(misses[["store_id", "brand", "address", "city", "state", "zip"]].to_string(index=False))


if __name__ == "__main__":
    main()

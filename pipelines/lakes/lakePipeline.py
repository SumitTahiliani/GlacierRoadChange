# run_pipeline.py
from shapely.geometry import Point
import geopandas as gpd
from extractLakes import extract_lakes, OUT_DIR
import pandas as pd

# AOI (Kedarnath demo)
lat, lon, rad_km =
aoi = Point(lon, lat).buffer(rad_km / 111.0)   # EPSG:4326

# Extract 2018 & 2023
l18 = extract_lakes(aoi, "2018-07-01/2018-07-31", tag="2018")
l23 = extract_lakes(aoi, "2023-07-01/2023-07-31", tag="2023")

# Simple change overlay (new or expanded lakes)
new_lakes = gpd.overlay(l23, l18, how="difference")
new_lakes["change"] = "gain_2018‑23"

# Concatenate & save master layer
combo = gpd.GeoDataFrame(pd.concat([l18, l23, new_lakes], ignore_index=True),
                         crs=l18.crs)
combo.to_file(OUT_DIR / "lakes_2018_2023_combo.geojson", driver="GeoJSON")
print("✅  Extraction & change layer saved to lakes_2018_2023_combo.geojson")

# Quick metrics
area18 = l18.to_crs(6933).area.sum() / 1e6
area23 = l23.to_crs(6933).area.sum() / 1e6
gain   = new_lakes.to_crs(6933).area.sum() / 1e6
print(f"Area 2018: {area18:.3f} km²\nArea 2023: {area23:.3f} km²\nGain: {gain:.3f} km²")

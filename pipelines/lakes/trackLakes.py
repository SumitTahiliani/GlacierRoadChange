#!/usr/bin/env python3
"""
track_lakes.py
--------------
Reads lakes_2018.geojson and lakes_2023.geojson (same CRS),
links polygons across years, outputs a CSV & merged GeoJSON
with lake_id, area_2018, area_2023, area_change_%.
"""

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

g18 = gpd.read_file("glacial_lakes_output/lakes_2018.geojson")
g23 = gpd.read_file("glacial_lakes_output/lakes_2023.geojson")

# ensure same CRS
if g18.crs != g23.crs:
    g23 = g23.to_crs(g18.crs)

# pre‑compute area in m² (equal‑area projection)
g18["area18"] = g18.to_crs(6933).area
g23["area23"] = g23.to_crs(6933).area

links = []
next_id = 1
for idx23, row23 in g23.iterrows():
    best_iou, best_idx18 = 0, None
    for idx18, row18 in g18.iterrows():
        inter = row23.geometry.intersection(row18.geometry).area
        union  = row23.geometry.union(row18.geometry).area
        iou = inter / union if union else 0
        if iou > best_iou:
            best_iou, best_idx18 = iou, idx18
    if best_iou >= 0.1:
        lake_id = best_idx18 + 1  # keep 2018 index as id
    else:
        lake_id = len(g18) + next_id; next_id += 1  # new lake
    links.append({"lake_id": lake_id,
                  "geometry": row23.geometry,
                  "area23": row23["area23"]})

# Merge info back to 2018 dataframe
track_df = pd.DataFrame(links)
merged = g18.rename(columns={"area18":"area18"}) \
           .merge(track_df[["lake_id","area23"]], left_index=True,
                  right_on="lake_id", how="outer")

merged["area_change_%"] = 100 * (merged["area23"] - merged["area18"]) / merged["area18"]
merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=g18.crs)
merged_gdf.to_file("glacial_lakes_output/lake_changes_2018_2023.geojson",
                   driver="GeoJSON")

print("lake_changes_2018_2023.geojson written with columns:",
      merged_gdf.columns.tolist())

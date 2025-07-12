#!/usr/bin/env python3
"""
evaluate_roads.py
-----------------
Compare vectorized model output (roads_vector.geojson)
to rasterized OSM mask (roads_mask_2023.tif).
Computes IoU, Precision, Recall, F1-score.
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

#Load original raster mask (ground truth)
with rasterio.open("road_data/roads_mask_2023.tif") as src:
    gt_mask = src.read(1).astype("uint8")     # shape: (H, W)
    transform = src.transform
    out_shape = (src.height, src.width)
    crs = src.crs

# Load vectorized model output and rasterize
gdf = gpd.read_file("road_output/roads_vector.geojson").to_crs(crs)

pred_mask = rasterize(
    [(geom.buffer(10), 1) for geom in gdf.geometry],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True
)

#Flatten masks and filter out background
y_true = gt_mask.flatten()
y_pred = pred_mask.flatten()
valid = (y_true + y_pred) > 0

y_true = y_true[valid]
y_pred = y_pred[valid]

#Compute metrics
iou = jaccard_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

print("Evaluation Metrics:")
print(f"  IoU        : {iou:.3f}")
print(f"  Precision  : {prec:.3f}")
print(f"  Recall     : {rec:.3f}")
print(f"  F1-score   : {f1:.3f}")
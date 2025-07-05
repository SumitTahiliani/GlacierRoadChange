#!/usr/bin/env python3
"""
vectorize_roads.py
------------------
Runs trained model on full Sentinel‑2 scene,
binarises, skeletonises, cleans spurs, outputs roads_vector.geojson
"""

from pathlib import Path
import rasterio
import itertools
import numpy as np, torch, torchvision
from skimage.morphology import binary_opening, skeletonize, remove_small_objects
from rasterio.features import shapes
import geopandas as gpd, shapely.geometry as sgeom, pandas as pd, networkx as nx

# paths
scene = rasterio.open("road_data/scene_2023.tif")
OUT   = Path("road_output"); OUT.mkdir(exist_ok=True)

# load model
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, num_classes=2)
model.load_state_dict(torch.load("models/roads_deeplab.pt", map_location="cpu"))
model.eval()

# ---- sliding‑window inference ----
window_size = 512
stride = 256
prob = np.zeros((scene.height, scene.width), dtype="float32")

for y in range(0, scene.height-window_size+1, stride):
    for x in range(0, scene.width-window_size+1, stride):
        win = rasterio.windows.Window(x, y, window_size, window_size)
        img = scene.read([1, 2, 3], window=win).astype("float32") / 10000  # (3, H, W)
        img_t = torch.tensor(img).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(img_t)["out"],1)[0,1].numpy()
        prob[y:y+window_size, x:x+window_size] = np.maximum(prob[y:y+window_size, x:x+window_size], p)

mask = (prob>0.5)
mask = binary_opening(mask, np.ones((3,3)))
mask = remove_small_objects(mask, 100)
skeleton = skeletonize(mask)

# ---- vectorise skeleton ----
records=[]
for geom,val in shapes(skeleton.astype("uint8"), mask=skeleton, transform=scene.transform):
    if val==1:
        line = sgeom.shape(geom)
        if line.length>20:                 # >20 m
            records.append({"geometry": line})

gdf = gpd.GeoDataFrame(pd.DataFrame(records), crs=scene.crs)
gdf.to_file(OUT/"roads_vector.geojson", driver="GeoJSON")
print("✓ roads_vector.geojson saved.")

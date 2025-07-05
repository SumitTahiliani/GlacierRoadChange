#!/usr/bin/env python3
"""
prep_road_data.py
-----------------
1. Download Sentinel‑2 L2A tile and clip to AOI.
2. Fetch OSM road lines (motorway→secondary) for same AOI.
3. Rasterise roads to 10 m mask aligning with Sentinel‑2.
4. Plot RGB image versus raster mask for sanity‑check.

Outputs
-------
road_data/scene_2023.tif        ← 4‑bandox RGBN
road_data/roads_mask_2023.tif   ← uint8 (1=road)
road_data/roads_mask_2023.png   ← preview PNG
"""

from pathlib import Path
import os
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rioxarray as rxr
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box
from mapminer.miners import Sentinel2Miner

# -------------------------------------------------------
# 1. AOI and date window (NH 44 south‑east of Hyderabad)
# -------------------------------------------------------
AOI_BOUNDS = (78.610, 17.200, 78.760, 17.350)  # (minLon, minLat, maxLon, maxLat)
DATE_RANGE = "2022-03-01/2023-03-30"           # post‑monsoon, low cloud
BANDS = ["B04", "B03", "B02", "B08"]           # R,G,B,NIR

aoi_geom = box(*AOI_BOUNDS)                    # shapely polygon
OUT_DIR = Path("road_data")
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# 2. Download Sentinel‑2 scene (mapminer)
# -------------------------------------------------------
miner = Sentinel2Miner()
ds_raw = miner.fetch(polygon=aoi_geom, daterange=DATE_RANGE)
ds = ds_raw.isel(time=0)[BANDS]                # first clear scene
# print("Sentinel‑2 dataset shape:", ds.shape)

# Save to COG for later use
scene_path = OUT_DIR / "scene_2023.tif"
if not scene_path.exists():
    ds.rio.to_raster(scene_path, tiled=True)
    print("Saved", scene_path)

# -------------------------------------------------------
# 3. Fetch OSM roads for same AOI
# -------------------------------------------------------
ox.settings.use_cache = True
tags = {"highway": ["motorway","trunk","primary","secondary",
                    "tertiary","residential","unclassified"]}
gdf_osm = ox.geometries_from_bbox(
    north=AOI_BOUNDS[3], south=AOI_BOUNDS[1],
    east=AOI_BOUNDS[2], west=AOI_BOUNDS[0],
    tags=tags
).to_crs(ds.rio.crs)   # reproject to UTM of Sentinel‑2 tile

buffer_m = 40        # metres
gdf_osm = gdf_osm.to_crs(ds.rio.crs)          # stay in metres
gdf_osm["geometry"] = gdf_osm.buffer(buffer_m)

print("OSM roads:", len(gdf_osm))

# -------------------------------------------------------
# 4. Rasterise roads to 10 m mask
# -------------------------------------------------------
height, width = ds.sizes["y"], ds.sizes["x"]
transform = ds.rio.transform()

shapes = ((geom, 1) for geom in gdf_osm.geometry)
mask = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True
)
from scipy.ndimage import binary_dilation
mask = binary_dilation(mask, iterations=2)   # was 2


mask_path = OUT_DIR / "roads_mask_2023.tif"
with rasterio.open(
    mask_path, "w",
    driver="GTiff",
    height=height, width=width,
    count=1, crs=ds.rio.crs, transform=transform,
    dtype="uint8"
) as dst:
    dst.write(mask, 1)
print("✓ Saved", mask_path)

# -------------------------------------------------------
# 5. Quick side‑by‑side plot
# -------------------------------------------------------
rgb = np.stack(
    [ds["B04"].values, ds["B03"].values, ds["B02"].values], axis=-1
) / 10000  # scale 0‑1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.imshow(rgb)
ax1.set_title("Sentinel‑2 RGB")
ax1.axis("off")

ax2.imshow(mask, cmap="gray")
ax2.set_title("Rasterised OSM roads")
ax2.axis("off")

plt.tight_layout()
png_path = OUT_DIR / "roads_mask_2023.png"
plt.savefig(png_path, dpi=150)
plt.show()
print("Preview saved to", png_path)

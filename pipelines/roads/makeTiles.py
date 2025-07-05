#!/usr/bin/env python3
"""
make_tiles.py
-------------
Reads road_data/scene_2023.tif  (4‑band image)
      road_data/roads_mask_2023.tif (uint8 mask)
Splits both into 256×256 tiles with 50 % overlap.
Saves .npz files: tiles/img_{idx}.npz, tiles/msk_{idx}.npz
"""

from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np

TILE = 96  
STRIDE = 48          # 50 % overlap
MIN_ROAD_PIX = 20    # skip empty tiles

src_img = rasterio.open("road_data/scene_2023.tif")
src_msk = rasterio.open("road_data/roads_mask_2023.tif")
out_dir = Path("tiles"); out_dir.mkdir(exist_ok=True)

idx = 0
for y in range(0, src_img.height - TILE + 1, STRIDE):
    for x in range(0, src_img.width - TILE + 1, STRIDE):
        win = Window(x, y, TILE, TILE)
        img = src_img.read(window=win).astype("float32") / 10000  # scale 0‑1
        msk = src_msk.read(1, window=win).astype("uint8")
        if msk.sum() < MIN_ROAD_PIX:
            continue
        # after loading each mask
        road_frac = msk.sum() / msk.size
        print(f"Tile {idx} road fraction: {road_frac:.4f}")

        np.savez_compressed(out_dir / f"img_{idx}.npz", img=img)
        np.savez_compressed(out_dir / f"msk_{idx}.npz", msk=msk)
        idx += 1

print(f"✓ Saved {idx} tile pairs in {out_dir}/")

from pathlib import Path
import numpy as np

bad_indices = []
for i in range(305):
    img_path = Path(f"tiles/img_{i}.npz")
    msk_path = Path(f"tiles/msk_{i}.npz")
    if not img_path.exists() or not msk_path.exists():
        continue
    img = np.load(img_path)["img"]
    msk = np.load(msk_path)["msk"]
    if img.shape[1:] != (96, 96) or msk.shape != (96, 96):
        print(f"Bad tile {i}: img {img.shape}, msk {msk.shape}")
        bad_indices.append(i)
        img_path.unlink()
        msk_path.unlink()
print(f"✓ Removed {len(bad_indices)} bad tiles.")

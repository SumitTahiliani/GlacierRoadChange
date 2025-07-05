import numpy as np, glob
masks = [np.load(p)["msk"] for p in glob.glob("tiles/msk_*.npz")]
fracs = [m.sum()/m.size for m in masks]
print("mean road %:", np.mean(fracs), "max:", np.max(fracs))
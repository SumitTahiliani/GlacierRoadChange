#!/usr/bin/env python3
"""
train_roads.py
--------------
Loads tiles/*.npz, trains DeepLabV3‑MobileNetV3 on 2 classes.
Saves best model weights to models/roads_deeplab.pt
"""

from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
Path("models").mkdir(exist_ok=True)

# ---------- dataset ----------
class RoadTiles(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = np.load(self.paths[i].with_name(f"img_{i}.npz"))["img"]
        msk = np.load(self.paths[i].with_name(f"msk_{i}.npz"))["msk"]
        return torch.tensor(img[:3]).float(), torch.tensor(msk).long()

tile_paths = sorted(Path("tiles").glob("img_*.npz"))
split = int(0.8*len(tile_paths))
train_ds = RoadTiles(tile_paths[:split])
val_ds   = RoadTiles(tile_paths[split:])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=8)

# ---------- model ----------
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, num_classes=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------- loss & optim ----------
pos_weight = torch.tensor([1.0]).to(device)   # 20× more weight on class 1
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # works on (N,1,H,W)
def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs*targets).sum(); union = probs.sum()+targets.sum()
    return 1 - (2*inter+eps)/(union+eps)  

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- training ----------
best_iou, EPOCHS = 0, 40
for epoch in range(EPOCHS):
    model.train()
    for img, msk in train_dl:
        img, msk = img.to(device), msk.to(device)
        out = model(img)["out"]
        logits = model(img)["out"][:,1:2]              # take road channel
        targets = (msk==1).float().unsqueeze(1)
        loss = bce(logits, targets) + dice_loss(logits, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # ---- validation ----
    model.eval(); inter=union=0
    with torch.no_grad():
        for img, msk in val_dl:
            img, msk = img.to(device), msk.to(device)
            pred = model(img)["out"].argmax(1)
            inter += ((pred==1)&(msk==1)).sum().item()
            union += ((pred==1)|(msk==1)).sum().item()
    iou = inter/union
    print(f"Epoch {epoch+1}/{EPOCHS}  IoU={iou:.3f}")
    if iou>best_iou:
        torch.save(model.state_dict(), "models/roads_deeplab.pt")
        best_iou = iou

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
import torch.nn.functional as F
Path("models").mkdir(exist_ok=True)
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tfms = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomShadow(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomFog(p=0.2),
    ToTensorV2(),
])

# ---------- dataset ----------
class RoadTiles(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img_path = self.paths[i]
        msk_path = img_path.with_name(img_path.name.replace("img_", "msk_"))

        img = np.load(img_path)["img"][:3]       # shape (3, 96, 96)
        msk = np.load(msk_path)["msk"]           # shape (96, 96)

        # to HWC for albumentations
        img = np.transpose(img, (1, 2, 0))        # (H, W, C)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 1.0
            msk = torch.tensor(msk).long()

        return img, msk


tile_paths = sorted(Path("tiles").glob("img_*.npz"))
split = int(0.8*len(tile_paths))
random.shuffle(tile_paths)  # shuffle before splitting
train_ds = RoadTiles(tile_paths[:split])
val_ds   = RoadTiles(tile_paths[split:])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=8)

# ---------- model ----------
model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None, num_classes=2
)
model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # if using RGB
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------- loss & optim ----------
pos_weight = torch.tensor([1.0]).to(device)   # 20× more weight on class 1
bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))  # if needed
def dice_loss(preds, targets, smooth=1e-7):
    probs = torch.sigmoid(preds)
    inter = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return 1 - (2 * inter + smooth) / (union + smooth) 

def combined_loss(logits, targets):
    return 0.7 * bce(logits, targets) + 0.3 * dice_loss(logits, targets)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- training ----------
best_iou, EPOCHS = 0, 150
for epoch in range(0, EPOCHS):
    model.train()
    for img, msk in train_dl:
        img, msk = img.to(device), msk.to(device).float()
        logits = model(img)["out"][:, 1:2]  # take road channel
        loss = combined_loss(logits, msk.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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

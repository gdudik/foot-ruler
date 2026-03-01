import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


class ShoeSeg(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            self.images += glob(os.path.join(images_dir, ext))
        self.images = sorted(self.images)
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.masks_dir, base + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        else:
            img = torch.from_numpy(img).permute(2,0,1).float()/255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    return (1.0 - (num / den)).mean()


@torch.no_grad()
def eval_iou(model, loader, device, thr=0.5):
    model.eval()
    inter = 0.0
    union = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = (torch.sigmoid(model(x)) > thr).float()
        inter += (p * y).sum().item()
        union += ((p + y) > 0).float().sum().item()
    return inter / (union + 1e-9)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    root = "Jumps Shoes.v2i.yolov8"
    train_images = os.path.join(root, "train", "images")
    train_masks  = os.path.join(root, "train", "masks")
    val_images   = os.path.join(root, "valid", "images")
    val_masks    = os.path.join(root, "valid", "masks")

    img_size = 512
    batch = 8
    epochs = 25
    lr = 3e-4
    thr = 0.5

    device = pick_device()
    print("Using device:", device)

    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    ds_tr = ShoeSeg(train_images, train_masks, transform=train_tf)
    ds_va = ShoeSeg(val_images, val_masks, transform=val_tf)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best = -1.0

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in tqdm(dl_tr, desc=f"epoch {ep}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = 0.5*bce(logits, y) + 0.5*dice_loss_from_logits(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()

        iou = eval_iou(model, dl_va, device, thr=thr)
        print(f"epoch {ep}: loss={running/len(dl_tr):.4f}  val IoU={iou:.4f}")

        if iou > best:
            best = iou
            torch.save(
                {
                    "arch": "unet",
                    "encoder": "efficientnet-b0",
                    "img_size": img_size,
                    "threshold": thr,
                    "state_dict": model.state_dict(),
                },
                "best_pytorch.pt"
            )
            print("  saved best_pytorch.pt")

    print("done. best IoU:", best)


if __name__ == "__main__":
    main()
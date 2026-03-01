import os
from glob import glob
import cv2
import numpy as np

def yolo_seg_to_mask(label_path: str, w: int, h: int) -> np.ndarray:
    """
    Reads a YOLO segmentation label file with normalized polygon points and returns
    a binary mask (uint8 0 or 255) of shape (h, w).
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask  # no label -> empty mask

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        # parts[0] is class id
        coords = list(map(float, parts[1:]))
        if len(coords) < 6 or (len(coords) % 2 != 0):
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            # clamp
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])

        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

    return mask

def convert_split(split_dir: str):
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    masks_dir  = os.path.join(split_dir, "masks")  # NEW

    os.makedirs(masks_dir, exist_ok=True)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        img_paths.extend(glob(os.path.join(images_dir, ext)))

    img_paths = sorted(img_paths)
    if not img_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        h, w = img.shape[:2]

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")
        mask = yolo_seg_to_mask(label_path, w, h)

        out_path = os.path.join(masks_dir, base + ".png")
        cv2.imwrite(out_path, mask)

    print(f"Created masks in: {masks_dir}")

def main():
    # point this at your exported dataset root folder
    dataset_root = "Jumps Shoes.v2i.yolov8"

    for split in ("train", "valid"):
        split_dir = os.path.join(dataset_root, split)
        if os.path.isdir(split_dir):
            convert_split(split_dir)

if __name__ == "__main__":
    main()
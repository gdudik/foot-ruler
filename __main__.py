import cv2
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description='Measure toe distance from board edge')
parser.add_argument('--orientation', '-o', choices=['left', 'right'], default='left',
                   help='Orientation for measurement: left measures from left edge, right measures from right edge')
parser.add_argument('--image', '-i', default='default.jpg',
                   help='Path to the image file to process')
parser.add_argument('--coords', '-c', default='board_coords.json',
                   help='Path to the board coordinates JSON file')
parser.add_argument('--board', '-b', type=int, default=1,
                   help='Board number to use for measurement (default: 1)')
args = parser.parse_args()



# === USER SETUP ===
# Path to your takeoff board image and coordinates file
IMAGE_PATH = args.image
ORIENTATION = args.orientation
COORDS_FILE = args.coords
BOARD_NUMBER = args.board

# Load board corner coordinates
with open(COORDS_FILE, 'r') as f:
    board_data = json.load(f)

# Extract coordinates for the specified board
board_key = str(BOARD_NUMBER)
if board_key not in board_data:
    available_boards = ', '.join(sorted(board_data.keys()))
    raise ValueError(f"Board {BOARD_NUMBER} not found in {COORDS_FILE}. Available boards: {available_boards}. Please run 'python calibrate.py -i {IMAGE_PATH} -b {BOARD_NUMBER}' to calibrate this board.")

board_pts = board_data[board_key]["coords"]
board_depth = board_data[board_key]["depth"]

# Real-world board dimensions (in centimeters)
BOARD_WIDTH_CM  = 122.0  # across runway (y-axis)

# Warp size: one pixel == 1 cm
WARP_WIDTH_PX  = int(board_depth)
WARP_HEIGHT_PX = int(BOARD_WIDTH_CM)



# Load image
orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Could not load image at '{IMAGE_PATH}'")

# Check if board coordinates file exists
if not os.path.exists(COORDS_FILE):
    raise FileNotFoundError(f"Calibration file '{COORDS_FILE}' not found. Please run 'python calibrate.py -i {IMAGE_PATH} -b {BOARD_NUMBER}' first.")







if len(board_pts) != 4:
    raise ValueError(f"Invalid board coordinates for board {BOARD_NUMBER} in {COORDS_FILE}. Expected 4 points, got {len(board_pts)}.")

print(f"Loaded board {BOARD_NUMBER} coordinates from {COORDS_FILE}")

# ---- Load PyTorch segmentation model for shoe detection ----
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ckpt = torch.load("best_pytorch.pt", map_location=DEVICE)
IMG_SIZE = int(ckpt.get("img_size", 512))
THRESH = float(ckpt.get("threshold", 0.5))
ENCODER = ckpt.get("encoder", "efficientnet-b0")

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

infer_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

print("Detecting shoe silhouette (PyTorch)...")

rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
x = infer_tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)  # [1,1,h,w]
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

mask_small = (prob > THRESH).astype(np.uint8)  # {0,1}

mask_resized = cv2.resize(
    mask_small,
    (orig.shape[1], orig.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# Keep only the biggest blob (helps prevent tiny false positives)
num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_resized, connectivity=8)
if num <= 1:
    raise RuntimeError("No shoe detected in the image. Cannot proceed with measurement.")
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask_resized = (labels == largest).astype(np.uint8)

# Find coordinates of all points in the mask
y_coords, x_coords = np.where(mask_resized > 0)
if len(x_coords) == 0:
    raise RuntimeError("No valid mask pixels found.")

# Find the extreme point based on orientation
if ORIENTATION == 'left':
    idx = np.argmin(x_coords)
    toe_pt = (int(x_coords[idx]), int(y_coords[idx]))
else:
    idx = np.argmax(x_coords)
    toe_pt = (int(x_coords[idx]), int(y_coords[idx]))

print(f"Toe tip detected at coordinates: {toe_pt}")

# === SORT CORNER POINTS AUTOMATICALLY ===
pts = np.array(board_pts, dtype='float32')
# sort by y to separate top vs bottom
y_sorted = pts[pts[:,1].argsort()]
top2 = y_sorted[:2]
bot2 = y_sorted[2:]
# within each, sort by x
tl, tr = top2[top2[:,0].argsort()]
bl, br = bot2[bot2[:,0].argsort()]
src = np.array([tl, tr, br, bl], dtype='float32')

# Destination rectangle for perspective
dst = np.array([[0,0],
                [WARP_WIDTH_PX, 0],
                [WARP_WIDTH_PX, WARP_HEIGHT_PX],
                [0, WARP_HEIGHT_PX]], dtype='float32')

# Compute homography
H, _ = cv2.findHomography(src, dst)
if H is None:
    raise RuntimeError("Homography computation failed.")
H_inv = np.linalg.inv(H)

# Warp toe point to board plane
toe_src = np.array([[toe_pt]], dtype='float32')
toe_warp = cv2.perspectiveTransform(toe_src, H)[0][0]

# Distance calculation based on orientation
if ORIENTATION == 'left':
    # Measure from left edge (x=0)
    distance_cm = toe_warp[0]
    edge_x = 0.0
    print(f"Distance from left edge: {distance_cm:.1f} cm")
else:  # right orientation
    # Measure from right edge (x=WARP_WIDTH_PX)
    distance_cm = WARP_WIDTH_PX - toe_warp[0]
    edge_x = WARP_WIDTH_PX
    print(f"Distance from right edge: {distance_cm:.1f} cm")

# Compute midpoint of the edge in warped frame and map back
mid_warp = np.array([[[edge_x, WARP_HEIGHT_PX / 2.0]]], dtype='float32')
mid_src = cv2.perspectiveTransform(mid_warp, H_inv)[0][0]
mid_pt = tuple(mid_src.astype(int))

# Compute intersection point on edge at same y-coordinate as toe
edge_warp = np.array([[[edge_x, toe_warp[1]]]], dtype='float32')  # Same y as toe
edge_src = cv2.perspectiveTransform(edge_warp, H_inv)[0][0]
edge_pt = tuple(edge_src.astype(int))

# Draw overlay and annotate
out = orig.copy()

# Add mask overlay for visualization
mask_overlay = np.zeros_like(out)
mask_overlay[mask_resized > 0] = [0, 255, 0]  # Green color for mask
out = cv2.addWeighted(out, 0.8, mask_overlay, 0.2, 0)  # Blend mask with original image

cv2.line(out, toe_pt, edge_pt, (255,255,255), 6, lineType=cv2.LINE_AA) #outline
cv2.line(out, toe_pt, edge_pt, (0,4,119), 2, lineType=cv2.LINE_AA)  # Main horizontal line

# Add vertical hash marks at each end
hash_length = 30  # Length of hash marks in pixels

# Calculate points for truly vertical hash marks
def get_hash_points(point):
    x, y = point
    return (
        (x, y - hash_length//2),  # Start point (up)
        (x, y + hash_length//2)   # End point (down)
    )

# Draw hash marks
edge_hash_start, edge_hash_end = get_hash_points(edge_pt)
toe_hash_start, toe_hash_end = get_hash_points(toe_pt)
cv2.line(out, edge_hash_start, edge_hash_end, (255,255,255), 6, lineType=cv2.LINE_AA)
cv2.line(out, toe_hash_start, toe_hash_end, (255,255,255), 6, lineType=cv2.LINE_AA)
cv2.line(out, edge_hash_start, edge_hash_end, (0,4,119), 2, lineType=cv2.LINE_AA)
cv2.line(out, toe_hash_start, toe_hash_end, (0,4,119), 2, lineType=cv2.LINE_AA)

# Calculate measurements and format labels
cm_label = f"{distance_cm:.1f} cm"
inches = distance_cm / 2.54  # Convert cm to inches
quarter = round(inches * 4) / 4  # Round to nearest quarter inch
inch_label = f"{quarter:.2f}\""  # Format as decimal with 2 places

# Combined label with newline
label = f"{cm_label} / {inch_label}"

# Generate output filename by adding "_annotated" before the extension
base_name, extension = os.path.splitext(IMAGE_PATH)
output_filename = f"{base_name}_annotated{extension}"

# Save result
cv2.imwrite(output_filename, out)
print(f"Saved {output_filename}")

base = Image.open(output_filename).convert("RGBA")
overlay = Image.open("bar.png").convert("RGBA")
pos_x = (base.width - overlay.width) // 2
pos_y = (math.floor(base.height * 0.8))
position = (pos_x, pos_y)
base.paste(overlay, position, overlay)

# Now add the text card on top of the bar overlay
draw = ImageDraw.Draw(base)

# Load font
font_size = 36
try:
    font = ImageFont.truetype("Arial Bold Italic.ttf", font_size)
except OSError:
    font = ImageFont.load_default()

# Calculate text position centered on the bar overlay
text_x = base.width // 2
text_y = pos_y + (overlay.height // 2)
text_pos = (text_x, text_y)

# Get text size to create background card (now for multiline text)
text_bbox = draw.textbbox(text_pos, label, font=font, anchor="mm")
padding = 8  # pixels of padding around text
card_bbox = (
    text_bbox[0] - padding,
    text_bbox[1] - padding,
    text_bbox[2] + padding,
    text_bbox[3] + padding
)


card_color = (255, 255, 255, 0) 
card_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
card_draw = ImageDraw.Draw(card_layer)
card_draw.rectangle(card_bbox, fill=card_color)
base = Image.alpha_composite(base, card_layer)

# Draw text on top
draw = ImageDraw.Draw(base)
draw.text(text_pos, label, font=font, fill=(255, 255, 255), anchor="mm")  # Note: PIL uses RGB

# Add warning box at top left
warning_text = "Measurement Generated by AthleticAI: Not Official"
warning_font_size = 20
try:
    warning_font = ImageFont.truetype("Arial Bold Italic.ttf", warning_font_size)
except OSError:
    warning_font = ImageFont.load_default()

# Position at top left with some margin
warning_margin = 10
warning_pos = (warning_margin, warning_margin)

# Get text size for warning box
warning_bbox = draw.textbbox(warning_pos, warning_text, font=warning_font)
warning_padding = 6
warning_card_bbox = (
    warning_bbox[0] - warning_padding,
    warning_bbox[1] - warning_padding,
    warning_bbox[2] + warning_padding,
    warning_bbox[3] + warning_padding
)

# Draw black background for warning
warning_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
warning_draw = ImageDraw.Draw(warning_layer)
warning_draw.rectangle(warning_card_bbox, fill=(0, 0, 0, 200))  # Black with alpha
base = Image.alpha_composite(base, warning_layer)

# Draw white warning text
draw = ImageDraw.Draw(base)
draw.text(warning_pos, warning_text, font=warning_font, fill=(255, 255, 255))  # White text

base = base.convert("RGB")
base.save(output_filename)

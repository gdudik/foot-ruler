import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
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

# Real-world board dimensions (in centimeters)
BOARD_WIDTH_CM  = 122.0  # across runway (x-axis)
BOARD_LENGTH_CM =  20.0  # from foul line back into runway (y-axis)

# Warp size: one pixel == 1 cm
WARP_WIDTH_PX  = int(BOARD_LENGTH_CM)
WARP_HEIGHT_PX = int(BOARD_WIDTH_CM)

# Load image
orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Could not load image at '{IMAGE_PATH}'")

# Check if board coordinates file exists
if not os.path.exists(COORDS_FILE):
    raise FileNotFoundError(f"Calibration file '{COORDS_FILE}' not found. Please run 'python calibrate.py -i {IMAGE_PATH} -b {BOARD_NUMBER}' first.")

# Load board corner coordinates
with open(COORDS_FILE, 'r') as f:
    board_data = json.load(f)

# Extract coordinates for the specified board
board_key = str(BOARD_NUMBER)
if board_key not in board_data:
    available_boards = ', '.join(sorted(board_data.keys()))
    raise ValueError(f"Board {BOARD_NUMBER} not found in {COORDS_FILE}. Available boards: {available_boards}. Please run 'python calibrate.py -i {IMAGE_PATH} -b {BOARD_NUMBER}' to calibrate this board.")

board_pts = board_data[board_key]

if len(board_pts) != 4:
    raise ValueError(f"Invalid board coordinates for board {BOARD_NUMBER} in {COORDS_FILE}. Expected 4 points, got {len(board_pts)}.")

print(f"Loaded board {BOARD_NUMBER} coordinates from {COORDS_FILE}")

# Load YOLO segmentation model for shoe detection
model = YOLO('best.pt')

# Run YOLO inference to detect the shoe silhouette
print("Detecting shoe silhouette...")
results = model(orig)

# Check if any shoes were detected
if len(results[0].masks) > 0:
    # Get the mask from the highest confidence detection
    mask = results[0].masks.data[0].cpu().numpy()
    
    # Resize mask to match original image dimensions
    mask_resized = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Find coordinates of all points in the mask
    y_coords, x_coords = np.where(mask_resized > 0)
    
    if len(x_coords) == 0:
        raise RuntimeError("No valid mask pixels found.")
    
    # Find the extreme point based on orientation
    if ORIENTATION == 'left':
        # For left orientation, find the leftmost point (minimum x)
        idx = np.argmin(x_coords)
        toe_pt = (int(x_coords[idx]), int(y_coords[idx]))
    else:
        # For right orientation, find the rightmost point (maximum x)
        idx = np.argmax(x_coords)
        toe_pt = (int(x_coords[idx]), int(y_coords[idx]))
    
    print(f"Toe tip detected at coordinates: {toe_pt}")
else:
    raise RuntimeError("No shoe detected in the image. Cannot proceed with measurement.")

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

cv2.line(out, toe_pt, edge_pt, (0,4,119), 2)  # Main horizontal line

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
cv2.line(out, edge_hash_start, edge_hash_end, (0,4,119), 2)
cv2.line(out, toe_hash_start, toe_hash_end, (0,4,119), 2)

# Calculate measurements and format labels
cm_label = f"{distance_cm:.1f} cm"
inches = distance_cm / 2.54  # Convert cm to inches
quarter = round(inches * 4) / 4  # Round to nearest quarter inch
inch_label = f"{quarter:.2f}\""  # Format as decimal with 2 places

# Combined label with newline
label = f"{cm_label}\n{inch_label}"

# Calculate midpoint for label with more offset for two lines
offset_y = 30  # Increased offset for two lines of text
mid_x = (toe_pt[0] + edge_pt[0]) // 2
mid_y = (toe_pt[1] + edge_pt[1]) // 2 + offset_y
mid = (mid_x, mid_y)

# Convert from OpenCV to PIL format
out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(out_rgb)
draw = ImageDraw.Draw(pil_img)

# Load font
font_size = 36
try:
    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
except OSError:
    font = ImageFont.load_default()

# Get text size to create background card (now for multiline text)
text_bbox = draw.textbbox(mid, label, font=font)
padding = 8  # pixels of padding around text
card_bbox = (
    text_bbox[0] - padding,
    text_bbox[1] - padding,
    text_bbox[2] + padding,
    text_bbox[3] + padding
)

# Draw white background card with slight transparency
card_color = (255, 255, 255, 230)  # White with alpha
card_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
card_draw = ImageDraw.Draw(card_layer)
card_draw.rectangle(card_bbox, fill=card_color)
pil_img = Image.alpha_composite(pil_img.convert('RGBA'), card_layer)

# Draw text on top
draw = ImageDraw.Draw(pil_img)
draw.text(mid, label, font=font, fill=(119, 4, 0))  # Note: PIL uses RGB

# Convert back to OpenCV format (convert from RGBA to RGB first)
pil_img = pil_img.convert('RGB')
out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Generate output filename by adding "_annotated" before the extension
base_name, extension = os.path.splitext(IMAGE_PATH)
output_filename = f"{base_name}_annotated{extension}"

# Save result
cv2.imwrite(output_filename, out)
print(f"Saved {output_filename}")

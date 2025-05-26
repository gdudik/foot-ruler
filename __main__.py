import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# === USER SETUP ===
# Path to your takeoff board image
IMAGE_PATH = 'takeoff.jpg'  # <-- change to your file

# Real-world board dimensions (in centimeters)
BOARD_WIDTH_CM  = 122.0  # across runway (x-axis)
BOARD_LENGTH_CM =  20.0  # from foul line back into runway (y-axis)

# Warp size: one pixel == 1 cm
WARP_WIDTH_PX  = int(BOARD_LENGTH_CM)
WARP_HEIGHT_PX = int(BOARD_WIDTH_CM)

# Storage for points
board_pts = []  # click any 4 corners of the board (order does not matter)
toe_pt = None   # click athlete's toe here

# Load image
orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Could not load image at '{IMAGE_PATH}'")
display = orig.copy()

# Mouse callback for selecting board corners and toe
mode = 'select_board'  # toggles between 'select_board' and 'select_toe'
def on_mouse(event, x, y, flags, _):
    global board_pts, toe_pt, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'select_board' and len(board_pts) < 4:
            board_pts.append([x, y])
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', display)
            if len(board_pts) == 4:
                print("4 corners selected. Now click the toe.")
                mode = 'select_toe'
                cv2.putText(display, 'Now click the toe', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Image', display)
        elif mode == 'select_toe' and toe_pt is None:
            toe_pt = (x, y)
            cv2.circle(display, toe_pt, 7, (0,0,255), -1)
            cv2.destroyAllWindows()
            cv2.setMouseCallback('Image', lambda *args: None)  # Disable further mouse input

# Show image and set callback
cv2.imshow('Image', display)
cv2.setMouseCallback('Image', on_mouse)
cv2.waitKey(0)  # Wait for both corners and toe to be selected

# Validate selections
if len(board_pts) != 4 or toe_pt is None:
    raise RuntimeError("Board corners or toe not fully selected.")

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

# Distance from left edge (x=0) in cm
# Use the warped x-coordinate for horizontal measurement
distance_cm = toe_warp[0]  # Remove the subtraction from BOARD_WIDTH_CM
print(f"Distance from left edge: {distance_cm:.1f} cm")

# Compute midpoint of the left edge in warped frame and map back
# Left edge corresponds to x=0, y midpoint at WARP_HEIGHT_PX/2
left_mid_warp = np.array([[[0.0, WARP_HEIGHT_PX / 2.0]]], dtype='float32')
left_mid_src = cv2.perspectiveTransform(left_mid_warp, H_inv)[0][0]
left_mid_pt = tuple(left_mid_src.astype(int))

# Compute intersection point on left edge at same y-coordinate as toe
left_edge_warp = np.array([[[0.0, toe_warp[1]]]], dtype='float32')  # Same y as toe, x=0
left_edge_src = cv2.perspectiveTransform(left_edge_warp, H_inv)[0][0]
left_edge_pt = tuple(left_edge_src.astype(int))

# Draw overlay and annotate
out = orig.copy()
cv2.line(out, toe_pt, left_edge_pt, (0,4,119), 2)  # Main horizontal line

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
left_hash_start, left_hash_end = get_hash_points(left_edge_pt)
right_hash_start, right_hash_end = get_hash_points(toe_pt)
cv2.line(out, left_hash_start, left_hash_end, (0,4,119), 2)
cv2.line(out, right_hash_start, right_hash_end, (0,4,119), 2)

# Calculate midpoint for label (existing code)
label = f"{distance_cm:.1f} cm"
mid = ((toe_pt[0]+left_edge_pt[0])//2, (toe_pt[1]+left_edge_pt[1])//2)

# Convert from OpenCV to PIL format
out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(out_rgb)
draw = ImageDraw.Draw(pil_img)

# Load a font (use a system font or provide path to a .ttf file)
font_size = 36
try:
    # Try to use Arial font (common on Mac)
    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
except OSError:
    # Fallback to default font
    font = ImageFont.load_default()

# Add text with PIL
draw.text(mid, label, font=font, fill=(119,4,0))  # Note: PIL uses RGB

# Convert back to OpenCV format
out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Save result
cv2.imwrite('annotated_jump.jpg', out)
print("Saved annotated_jump.jpg")

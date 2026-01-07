import cv2
import numpy as np
import json
import os
import sys
import argparse

# File to save coordinates
COORDS_FILE = 'board_coords.json'

def save_coordinates(coords, board_number, file_path=COORDS_FILE):
    """Save coordinates to a JSON file with support for multiple boards"""
    # Load existing data if file exists
    board_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
                # Handle backward compatibility - convert old format to new format
                if isinstance(existing_data, list) and len(existing_data) == 4:
                    # Old format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Convert to new format and assume it was board 1
                    board_data = {"1": existing_data}
                    print("Converted old format coordinates to new multi-board format")
                elif isinstance(existing_data, dict):
                    # New format: {"1": [[x1,y1], ...], "2": [[x1,y1], ...]}
                    board_data = existing_data
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not read existing coordinates file, starting fresh")
            board_data = {}
    
    # Add/update the current board's coordinates
    board_data[str(board_number)] = coords
    
    # Save updated data
    with open(file_path, 'w') as f:
        json.dump(board_data, f, indent=2)
    print(f"Board {board_number} coordinates saved to {file_path}")

def on_mouse(event, x, y, flags, params):
    """Mouse callback for selecting board corners"""
    img, coords, board_number = params
    display_img = img.copy()
    
    # Show all existing points
    for i, point in enumerate(coords):
        cv2.circle(display_img, (point[0], point[1]), 5, (0, 255, 0), -1)
        cv2.putText(display_img, str(i+1), (point[0]+10, point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add new point on click
    if event == cv2.EVENT_LBUTTONDOWN and len(coords) < 4:
        coords.append([x, y])
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display_img, str(len(coords)), (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"Added point {len(coords)}: ({x}, {y})")
        
        # If we have all 4 points, save and close window automatically
        if len(coords) == 4:
            print("All 4 corners selected!")
            save_coordinates(coords, board_number)
            # Signal window to close
            cv2.destroyWindow("Calibrate - Select 4 Board Corners")
            
    cv2.imshow("Calibrate - Select 4 Board Corners", display_img)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calibrate board coordinates by selecting corners',
        epilog='Example: python calibrate.py -i jump_image.jpg -b 1'
    )
    parser.add_argument('--image', '-i', required=True,
                       help='Path to the image file to calibrate (required)')
    parser.add_argument('--board', '-b', type=int, required=True,
                       help='Board number to calibrate, must be 1 or greater (required)')
    args = parser.parse_args()
    
    # Validate board number
    if args.board < 1:
        print("Error: Board number must be 1 or greater.")
        return 1
    
    image_path = args.image
    board_number = args.board
    
    print(f"Calibrating board {board_number}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return 1
        
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return 1
        
    # Initialize coordinates list
    coords = []
    
    # Create window explicitly with normal properties
    cv2.namedWindow("Calibrate - Select 4 Board Corners", cv2.WINDOW_NORMAL)
    
    # Show the image in the window
    cv2.imshow("Calibrate - Select 4 Board Corners", img)
    
    # Force an initial redraw
    cv2.waitKey(1)
    
    # Set mouse callback
    cv2.setMouseCallback("Calibrate - Select 4 Board Corners", on_mouse, [img, coords, board_number])
    
    print("Click on the 4 corners of the takeoff board.")
    print("The order doesn't matter - they'll be sorted automatically later.")
    
    # Wait for user to select points or exit
    while True:
        key = cv2.waitKey(100)
        if key != -1:  # Any key pressed
            break
            
        # Check if window is still open
        if cv2.getWindowProperty("Calibrate - Select 4 Board Corners", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed")
            break
            
        # Exit loop immediately if we have all points
        if len(coords) == 4:
            break
    
    cv2.destroyAllWindows()
    
    if len(coords) == 4:
        print("Calibration complete!")
        return 0
    else:
        print("Calibration cancelled.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

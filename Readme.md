# Installation
- This project requires Python 3.11. Highly recommend using the Python install manager which works similarly to NVM. 
- This project requires [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
- For Windows, you'll need to install Microsoft Visual C++ Redistributable on your machine. Use `winget install -e --id Microsoft.VCRedist.2015+.x64`.
- Clone the repo to a folder
- Run `poetry install` to set up a venv and install all other dependencies

# Usage
1. Calibrate your takeoff boards
   1. Set up and align your takeoff board camera(s).
   2. Measure (in cm) the depth of your takeoff board from the perspective of the athlete (athlete's Z-axis). Standard sunken boards (like those available from Gill) are 20cm. If your officials are putting down a taped board, encourage them to make the depth of the board a whole cm number (18.0 or 19.0, not 18.5). Adding a layer of tape to get there can help. The closer you are to a whole cm, the more accurate the measurements produced will be. 
   3. Export a sample image from each board you're running. This is to define the corners of the boards for the computer vision warp module.
   4. For each board, run `poetry run calibrate`. The following flags are available
      - `--help` or `-h`: Explains all of this.
      - `--image` or `-i`: Path to the calibration image you exported above (required)
      - `--board` or `-b`: Board number to calibrate (required). If you are running multiple takeoff boards, you'll need to ensure that board 1 is window 1 in Lynx, board 2 is windows 2, etc.
      - `--depth` or `-d`: Depth (in the direction of athlete travel) of the takeoff board, in whole centimeters, as measured in 1.ii. If not defined, this will default to 20
      - `--orientation` or `-o`: Direction of athlete travel relative to the camera. Enter `left` or `right`.
   5. When you run the command, a window will appear with the image you selected. Click the four corners of the takeoff board in any order. Resize the window if necessary to get a good view. The more accurate you are, the more accurate your measurements will end up being. When you click the final point, the window will disappear. Board coordinates, depth, and orientation information are stored in `board_coords.json`
2. To test a sample image with a foot in it, run `poetry run measure`. The following flags are available:
   - `--help` or `-h`: Explains the following.
   - `--image` or `-i`: Path to the image file to process (required).
   - `--coords` or `-c`: Path to the coordinates file. Defaults to `./board_coords.json`.
   - `--path` or `-p`: Path for image export. Defaults to the current folder.
3. To start the tool for automated processing, run `poetry run start`. The following flags are available:
   - `--help` or `-h`: Explains the following.
   - `--path` or `-p`: Path to the raw images. This folder will be watched for new images. When a new image is detected, the image will be processed.
   - `--output` or `-o`: Location to output processed images. After processing, annotated images will be placed in this folder with the same file name as they had pre-processing.
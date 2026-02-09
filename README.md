# LAYZER: Leaf Area analYZER

<img width="1485" height="998" alt="image" src="https://github.com/user-attachments/assets/06cf5a3d-beb8-4202-a0a1-3f9ef731cc89" />

A Python-based desktop application for calculating the surface area of green leaves from images. This tool uses computer vision techniques (ExG index and Otsu's thresholding) to automatically detect green regions and calculate their area in square centimeters.

## Features

- **Automated Detection**: Uses Excess Green (ExG) index to isolate green plant matter.
- **Auto-Thresholding**: Integrated Otsu's method for automatic binary thresholding, with manual override.
- **ROI Selection**: Draw a polygon Region of Interest to exclude unwanted background elements.
- **Real-time Calibration**: Set the scale (pixels per cm) dynamically using a reference object in the image.
- **Pan & Zoom**: Inspect high-resolution images with mouse wheel zoom and middle-button panning.
- **Batch Processing**: Navigate through multiple images in a folder.
- **Result Export**: Save calculations to a CSV file.

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
   *Or use the provided `LAYZER.exe` if available.*

## Usage Guide

### 1. Open Images
Click **"1. Open Files"** to select one or more images (`.jpg`, `.png`, `.tif`, etc.). The first image will be displayed in the viewer.

### 2. Set Scale
To convert pixels to centimeters, you must define a known length in the image (e.g., a ruler included in the photo).
1. Click **"2. Set Scale"**.
2. Click two points on the image corresponding to a known distance.
3. Enter the real-world distance in **cm** in the popup dialog.
   - *The scale is now set and displayed in the sidebar.*

### 3. Draw ROI (Optional)
If there is noise or other green objects in the background:
1. Click **"3. Draw ROI (Polygon)"**.
2. Click to place points around the leaf you want to measure.
3. **Right-click** to close the polygon and finish.
   - *Only pixels inside this region will be counted.*
   - Click **"Reset ROI"** to clear the selection.

### 4. Threshold Adjustment
The app tries to detect the leaf automatically using **Auto (Otsu)**.
- If the detection is inaccurate, uncheck "Auto (Otsu)" or move the **slider** / click **+/- buttons** to adjust the sensitivity manually.
- The green overlay shows exactly what is being measured.

### 5. Calculate & Save
1. The **Area** is displayed in real-time in the sidebar.
2. Click **"Save Results"** to append the current measurement to `results.csv`.
3. Use **"Next >"** / **"< Prev"** to proceed to other images.

## Controls

| Action | Control |
|--------|---------|
| **Zoom In/Out** | Mouse Wheel |
| **Pan (Move)** | Middle Mouse Button Drag (Wheel Click) |
| **Draw/Select** | Left Click |
| **Finish ROI** | Right Click |


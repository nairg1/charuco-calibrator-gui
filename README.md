<<<<<<< HEAD
# charuco-calibrator-gui
Desktop GUI for ChArUco camera calibration with intrinsics, single-camera extrinsics, multi-camera extrinsics, and 3D visualization.
=======
# ChArUco Calibrator GUI

This is the public-safe version of the desktop calibration tool:

- `charuco_calibrator_gui.py`

The folder is organized in a more package-style layout for GitHub:

```text
charuco_calibrator_public/
├── charuco_calibrator_gui.py
├── requirements.txt
├── README.md
├── .gitignore
├── docs/
│   └── images/
└── src/
    └── charuco_calibrator/
        ├── __init__.py
        ├── __main__.py
        └── gui.py
```

It supports:

- intrinsics calibration from a folder of ChArUco board images
- single-camera extrinsics estimation from one image or a folder
- multi-camera relative extrinsics from one synchronized image per camera
- visualization of saved extrinsics JSON output

## Install

```bash
python3 -m pip install -r requirements.txt
```

Important:

- You need `opencv-contrib-python`, not just `opencv-python`
- Square length and marker length must use the same unit
- For multi-camera extrinsics, all camera images should show the same board pose at the same moment

## Run

```bash
python3 charuco_calibrator_gui.py
```

You can also run it as a package:

```bash
PYTHONPATH=src python3 -m charuco_calibrator
```

## Portable Defaults

- The app no longer depends on user-specific absolute paths
- Output files default to `charuco_calibration_output/` next to the script
- Input image and folder paths start blank so each user can choose their own data
- Multi-camera rows start with generic names like `cam1`, `cam2`, and can be resized to any camera count

## Screenshots

### Intrinsics

![Intrinsics tab](docs/images/gui-intrinsics.png)

### Single-Camera Extrinsics

![Single-camera extrinsics tab](docs/images/gui-single-camera-extrinsics.png)

### Multi-Camera Extrinsics

![Multi-camera extrinsics tab](docs/images/gui-multi-camera.png)

### Visualizer

![Visualizer tab](docs/images/gui-visualizer.png)

## Typical Workflow

### 1. Intrinsics

- Open the `Intrinsics` tab
- Set your board parameters
- Choose a folder of ChArUco frames for one camera
- Save the output as `<camera_name>_intrinsics.json`

### 2. Single-Camera Extrinsics

- Open the `Single-Camera Extrinsics` tab
- Load that camera's intrinsics JSON
- Choose one image or a folder of ChArUco frames
- Save the output JSON

### 3. Multi-Camera Extrinsics

- Open the `Multi-Camera Extrinsics` tab
- Set `Camera Count` and click `Apply Count`
- Enable one row per camera you want to solve
- For each enabled row, choose that camera's intrinsics JSON and one synchronized ChArUco image
- Pick a reference camera
- Run the solve and save the JSON
>>>>>>> 7f87021 (Initial commit)

# Dual Camera Triangulation

Real-time triangulation of hand and body keypoints using two synchronized cameras. The system uses OpenCV for capture/calibration and MediaPipe for keypoint detection, then triangulates 3D positions and visualizes them.

## Project Structure

- **src/**: Main application code.
  - **calibration/**
    - **estimate_extrinsics_aruco.py**: Estimate camera extrinsics (rotation/translation) using ArUco markers.
  - **capture/**
    - **list_cameras.py**: List available camera devices and their indices.
    - **dual_cam_preview.py**: Preview two cameras side-by-side to verify alignment and sync.
  - **config/**
    - **config.yaml**: Camera/detector settings and thresholds.
  - **detection/**
    - **mediapipe_adapter.py**: MediaPipe hand detection wrapper.
    - **mediapipe_pose_adapter.py**: MediaPipe body pose detection wrapper.
    - **hand_keypoints.py**: Hand keypoint processing utilities.
    - **download_hand_landmarker_model.py**: Download the hand landmark model.
    - **download_pose_landmarker_model.py**: Download the body pose model.
  - **markers/**
    - **aruco_pose.py**: ArUco marker detection and camera pose estimation helpers.
  - **triangulation/**
    - **rays.py**: Generate camera rays from pixels using intrinsics/extrinsics.
    - **triangulate.py**: Triangulation math utilities.
    - **dual_live_triangulate.py**: Live dual-camera hand triangulation runner.
    - **dual_live_triangulate_body.py**: Live dual-camera body pose triangulation runner.
  - **visualization/**
    - **viewer.py**, **pg_viewer.py**: Visualize 3D points and camera feeds.
    - **hand_skeleton.py**, **body_skeleton.py**: Skeleton definitions for rendering.
  - **io/**
    - **loader.py**: Load camera calibration and model assets.
    - **writer.py**: Save triangulated outputs.
  - **types/**
    - **index.py**: Data structures for keypoints and camera parameters.

- **data/**
  - **calibration/**
    - **camera1/**, **camera2/**: `camera_matrix.txt`, `distortion_coefficients.txt` for each camera.
    - **extrinsics.npz**: Output of extrinsics estimation (R, t for camera pair).

- **models/**
  - **hand_landmarker.task**, **pose_landmarker_full.task**: MediaPipe model files.

- **requirements.txt**: Python dependencies.

## Setup (Windows)

1) Create a virtual environment (optional, recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Download models (if not already in `models/`):

```powershell
python -m src.detection.download_hand_landmarker_model
python -m src.detection.download_pose_landmarker_model
```

## Camera Setup and Calibration

1) Identify camera indices:

```powershell
python -m src.capture.list_cameras
```

2) Preview both cameras to align framing:

```powershell
python -m src.capture.dual_cam_preview <cam1_id> <cam2_id>
```

3) Intrinsics: Place `camera_matrix.txt` and `distortion_coefficients.txt` in `data/calibration/camera1/` and `camera2/`.

4) Extrinsics (relative pose between cameras):

```powershell
python -m src.calibration.estimate_extrinsics_aruco <cam1_id> <cam2_id>
```

This writes `extrinsics.npz` to `data/calibration/`.

## Running

Hand triangulation (MediaPipe Hands):

```powershell
python -m src.triangulation.dual_live_triangulate <cam1_id> <cam2_id>
```

Body pose triangulation (MediaPipe Pose):

```powershell
python -m src.triangulation.dual_live_triangulate_body <cam1_id> <cam2_id>
```

Adjust detection and visualization settings in `src/config/config.yaml`.

## Notes

- Ensure both cameras share similar exposure and frame rate for consistent triangulation.
- If MediaPipe fails to load models, confirm files exist in `models/` or re-run the download scripts.
- On Windows, if camera access fails, check privacy settings and close other apps using the camera.

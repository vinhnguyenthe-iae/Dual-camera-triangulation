import sys
import cv2
import numpy as np

from src.io.loader import load_camera_calibration
from src.detection.mediapipe_pose_adapter import MediaPipePoseAdapter
from src.triangulation.triangulate import triangulate_points
from src.visualization.pg_body_viewer import PGBodyViewer
from src.markers.aruco_pose import detect_aruco_markers, estimate_camera_pose


def open_cap(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cap


def get_first_body_keypoints(adapter, frame):
    pts = adapter.get_first_body_keypoints_px(frame)
    if pts is None:
        return None
    return np.array(pts, dtype=np.float32)


def main():
    cam1 = 0
    cam2 = 3
    if len(sys.argv) >= 3:
        try:
            cam1 = int(sys.argv[1])
            cam2 = int(sys.argv[2])
        except ValueError:
            print("Usage: python src/triangulation/dual_live_triangulate_body.py <index1> <index2>")
            sys.exit(1)

    # Load intrinsics
    K1, D1 = load_camera_calibration('data/calibration/camera1')
    K2, D2 = load_camera_calibration('data/calibration/camera2')

    # Load extrinsics
    try:
        extr = np.load('data/calibration/extrinsics.npz')
        R = extr['R']
        t = extr['t']
    except Exception as e:
        print("Failed to load extrinsics (data/calibration/extrinsics.npz). Run the ArUco estimator first.")
        print(e)
        sys.exit(1)

    cap1 = open_cap(cam1)
    cap2 = open_cap(cam2)
    if not cap1.isOpened() or not cap2.isOpened():
        print("Failed to open one or both cameras.")
        sys.exit(1)

    detector = MediaPipePoseAdapter()
    viewer = PGBodyViewer()

    try:
        while True:
            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if not ok1 or not ok2:
                print("Failed to read frames.")
                break

            pts1 = get_first_body_keypoints(detector, f1)
            pts2 = get_first_body_keypoints(detector, f2)

            if pts1 is not None and pts2 is not None and pts1.shape[0] == pts2.shape[0]:
                pts3d = triangulate_points(pts1, pts2, K1, K2, R, t)
                viewer.update(pts3d)

            # Detect ArUco marker in camera1 view and update axes
            corners, ids = detect_aruco_markers(f1)
            if ids is not None and len(ids) > 0:
                rvecs, tvecs = estimate_camera_pose(corners, K1, D1)
                if rvecs is not None and tvecs is not None:
                    rvec = rvecs[0].reshape(3)
                    tvec = tvecs[0].reshape(3)
                    Rm, _ = cv2.Rodrigues(rvec)
                    axis_len = 0.1  # meters; match MARKER_LENGTH for consistency
                    x_end = tvec + Rm @ np.array([axis_len, 0, 0])
                    y_end = tvec + Rm @ np.array([0, axis_len, 0])
                    z_end = tvec + Rm @ np.array([0, 0, axis_len])
                    viewer.update_marker(tvec, {'x': x_end, 'y': y_end, 'z': z_end})

                    # Compute marker square corners in world (camera1) coords
                    s = 0.1
                    half = s / 2.0
                    corners_m = [
                        np.array([-half,  half, 0.0]),
                        np.array([ half,  half, 0.0]),
                        np.array([ half, -half, 0.0]),
                        np.array([-half, -half, 0.0]),
                    ]
                    corners_w = np.array([tvec + Rm @ c for c in corners_m])
                    viewer.update_marker_square(corners_w)

            viewer.process_events()
            # Update camera 3 preview
            viewer.update_cam2(f2)
            # Update camera 0 preview
            viewer.update_cam1(f1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.close()
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import sys
import numpy as np
import cv2

from src.io.loader import load_camera_calibration
from src.markers.aruco_pose import detect_aruco_markers, estimate_camera_pose


def rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def main():
    cam1_index = 0
    cam2_index = 3
    if len(sys.argv) >= 3:
        try:
            cam1_index = int(sys.argv[1])
            cam2_index = int(sys.argv[2])
        except ValueError:
            print("Usage: python src/calibration/estimate_extrinsics_aruco.py <cam1_index> <cam2_index>")
            sys.exit(1)

    # Load intrinsics
    K1, D1 = load_camera_calibration('data/calibration/camera1')
    K2, D2 = load_camera_calibration('data/calibration/camera2')

    # Open cameras
    cap1 = cv2.VideoCapture(cam1_index, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        cap1.release()
        cap1 = cv2.VideoCapture(cam1_index, cv2.CAP_MSMF)
    cap2 = cv2.VideoCapture(cam2_index, cv2.CAP_DSHOW)
    if not cap2.isOpened():
        cap2.release()
        cap2 = cv2.VideoCapture(cam2_index, cv2.CAP_MSMF)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Failed to open one or both cameras.")
        sys.exit(1)

    print("Show an ArUco marker to both cameras. Press 's' to sample, 'q' to quit.")
    R_c1_m = None
    t_c1_m = None
    R_c2_m = None
    t_c2_m = None

    while True:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 or not ok2:
            print("Failed to read frames.")
            break

        cv2.imshow(f"Cam {cam1_index}", f1)
        cv2.imshow(f"Cam {cam2_index}", f2)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # Detect marker in both frames
            corners1, ids1 = detect_aruco_markers(f1)
            corners2, ids2 = detect_aruco_markers(f2)

            if ids1 is None or ids2 is None:
                print("Marker not detected in both cameras. Try again.")
                continue

            # Choose a common marker id
            ids1_set = set(ids1.flatten().tolist())
            ids2_set = set(ids2.flatten().tolist())
            common = ids1_set.intersection(ids2_set)
            if not common:
                print("No common marker IDs between cameras. Try again.")
                continue

            # Use the first common id
            target_id = list(common)[0]

            # Filter corners for the target marker
            def select_corners(corners, ids, target):
                for i, mid in enumerate(ids.flatten().tolist()):
                    if mid == target:
                        return [corners[i]]
                return None

            c1 = select_corners(corners1, ids1, target_id)
            c2 = select_corners(corners2, ids2, target_id)

            if c1 is None or c2 is None:
                print("Failed to select marker corners. Try again.")
                continue

            rvecs1, tvecs1 = estimate_camera_pose(c1, K1, D1)
            rvecs2, tvecs2 = estimate_camera_pose(c2, K2, D2)

            if rvecs1 is None or rvecs2 is None:
                print("Pose estimation failed in one camera. Try again.")
                continue

            rvec1 = rvecs1[0].reshape(3)
            tvec1 = tvecs1[0].reshape(3)
            rvec2 = rvecs2[0].reshape(3)
            tvec2 = tvecs2[0].reshape(3)

            # estimatePoseSingleMarkers gives marker wrt camera (T_m_c)
            R_m_c1 = rodrigues_to_matrix(rvec1)
            R_m_c2 = rodrigues_to_matrix(rvec2)

            T_m_c1 = make_transform(R_m_c1, tvec1)
            T_m_c2 = make_transform(R_m_c2, tvec2)

            # Invert to get camera wrt marker (world): T_c_m
            T_c1_m = invert_transform(T_m_c1)
            T_c2_m = invert_transform(T_m_c2)

            # Relative transform from cam2 to cam1: T_c1c2 = T_c1_m * inv(T_c2_m)
            T_c2_m_inv = invert_transform(T_c2_m)
            T_c1c2 = T_c1_m @ T_c2_m_inv

            R_c1c2 = T_c1c2[:3, :3]
            t_c1c2 = T_c1c2[:3, 3]

            # Save extrinsics
            out_path = 'data/calibration/extrinsics.npz'
            np.savez(out_path, R=R_c1c2, t=t_c1c2)
            print(f"Saved extrinsics to {out_path}")
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

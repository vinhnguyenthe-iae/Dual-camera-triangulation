import cv2
import numpy as np

# OpenCV 4.12+ API: use getPredefinedDictionary and ArUcoDetector
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
MARKER_LENGTH = 0.15  # Length of the marker's side in meters

def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    return corners, ids

def estimate_camera_pose(corners, camera_matrix, distortion_coeffs):
    if corners is not None and len(corners) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, distortion_coeffs)
        return rvecs, tvecs
    return None, None

def draw_markers(frame, corners, ids):
    if corners is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
import os
import numpy as np

def load_camera_calibration(camera_folder):
    """Load camera calibration parameters from the specified folder."""
    camera_matrix_path = os.path.join(camera_folder, 'camera_matrix.txt')
    distortion_coeffs_path = os.path.join(camera_folder, 'distortion_coefficients.txt')
    
    if not os.path.exists(camera_matrix_path) or not os.path.exists(distortion_coeffs_path):
        raise FileNotFoundError("Camera calibration files not found.")
    
    camera_matrix = np.loadtxt(camera_matrix_path).reshape(3, 3)
    distortion_coeffs = np.loadtxt(distortion_coeffs_path).reshape(-1, 1)
    
    return camera_matrix, distortion_coeffs

def load_keypoints(keypoint_file):
    """Load hand keypoints from a specified text file."""
    keypoints = {}
    
    with open(keypoint_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            frame_num = int(parts[0])
            keypoints[frame_num] = []
            for i in range(21):
                x = float(parts[1 + i * 2])
                y = float(parts[2 + i * 2])
                keypoints[frame_num].append((x, y))
    
    return keypoints
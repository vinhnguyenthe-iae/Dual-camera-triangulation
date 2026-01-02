import numpy as np

def triangulate_points(points_2d_cam1, points_2d_cam2, camera_matrix1, camera_matrix2, rotation_matrix, translation_vector):
    """
    Triangulate 3D points from 2D points in two camera views.

    Args:
        points_2d_cam1 (np.ndarray): 2D points from camera 1 (shape: Nx2).
        points_2d_cam2 (np.ndarray): 2D points from camera 2 (shape: Nx2).
        camera_matrix1 (np.ndarray): Camera matrix for camera 1 (shape: 3x3).
        camera_matrix2 (np.ndarray): Camera matrix for camera 2 (shape: 3x3).
        rotation_matrix (np.ndarray): Rotation matrix from camera 2 to camera 1 (shape: 3x3).
        translation_vector (np.ndarray): Translation vector from camera 2 to camera 1 (shape: 3x1).

    Returns:
        np.ndarray: 3D points in world coordinates (shape: Nx3).
    """
    assert points_2d_cam1.shape[0] == points_2d_cam2.shape[0], "Number of points must match in both cameras."

    # Create projection matrices
    projection_matrix1 = camera_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_matrix2 = camera_matrix2 @ np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))

    points_3d = []
    
    for p1, p2 in zip(points_2d_cam1, points_2d_cam2):
        A = np.array([
            p1[0] * projection_matrix1[2] - projection_matrix1[0],
            p1[1] * projection_matrix1[2] - projection_matrix1[1],
            p2[0] * projection_matrix2[2] - projection_matrix2[0],
            p2[1] * projection_matrix2[2] - projection_matrix2[1]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1]
        point_3d /= point_3d[-1]  # Normalize to get homogeneous coordinates
        points_3d.append(point_3d[:-1])  # Exclude the last coordinate

    return np.array(points_3d)
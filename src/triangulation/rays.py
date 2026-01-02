import numpy as np

def pixel_to_world_ray(px_2d, R_cw, t_cw, K, D):
    """Generates a 3D ray from a 2D pixel."""
    px_2d_dist = np.array([[px_2d]], dtype=np.float32)
    px_2d_norm_undist = cv2.undistortPoints(px_2d_dist, K, D, P=None)
    v_cam = np.array([px_2d_norm_undist[0, 0, 0], px_2d_norm_undist[0, 0, 1], 1.0])
    v_cam_dir = v_cam / np.linalg.norm(v_cam)
    v_world_dir = (R_cw @ v_cam_dir).reshape(3)
    ray_origin = t_cw.reshape(3)
    return ray_origin, v_world_dir

def triangulate_from_rays(rays):
    """Triangulates 3D point from multiple rays."""
    A = np.zeros((3, 3))
    b = np.zeros((3,))
    I = np.identity(3)
    
    for origin, direction in rays:
        O = origin.reshape(3)
        V = direction.reshape(3)
        V = V / np.linalg.norm(V)
        proj_matrix = I - np.outer(V, V)
        A += proj_matrix
        b += proj_matrix @ O
        
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(A) @ b

    total_error = 0
    for origin, direction in rays:
        O = origin.reshape(3)
        V = direction.reshape(3)
        V = V / np.linalg.norm(V)
        dist_vec = (P - O) - np.dot(P - O, V) * V
        total_error += np.linalg.norm(dist_vec)
    mean_error = total_error / len(rays)
    
    return P, mean_error
from typing import List, Tuple, Dict

# Define a type for hand keypoints
Keypoint = Dict[str, float]

# Define a type for a list of keypoints
KeypointsList = List[Keypoint]

# Define a type for camera parameters
CameraParameters = Tuple[List[float], List[float]]  # (camera_matrix, distortion_coefficients)

# Define a type for triangulated points
TriangulatedPoint = Tuple[float, float, float]  # (x, y, z)

# Define a type for a list of triangulated points
TriangulatedPointsList = List[TriangulatedPoint]
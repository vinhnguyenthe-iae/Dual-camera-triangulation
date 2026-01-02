import numpy as np

def extract_keypoints(detection_results):
    """Extract hand keypoints from detection results."""
    keypoints = []
    if detection_results.multi_hand_landmarks:
        for hand_landmarks in detection_results.multi_hand_landmarks:
            hand_keypoints = []
            for landmark in hand_landmarks.landmark:
                hand_keypoints.append((landmark.x, landmark.y, landmark.z))
            keypoints.append(hand_keypoints)
    return keypoints

def process_keypoints(keypoints):
    """Process extracted keypoints for triangulation."""
    processed_keypoints = []
    for hand_keypoints in keypoints:
        if hand_keypoints:
            processed_keypoints.append(np.array(hand_keypoints))
        else:
            processed_keypoints.append(None)
    return processed_keypoints

def keypoints_to_world_coordinates(keypoints, camera_matrix, distortion_coeffs):
    """Convert 2D keypoints to 3D world coordinates."""
    world_coordinates = []
    for kp in keypoints:
        if kp is not None:
            # Assuming a simple mapping for demonstration purposes
            # In practice, you would use triangulation or other methods
            world_coordinates.append(kp)  # Placeholder for actual conversion
        else:
            world_coordinates.append(None)
    return world_coordinates
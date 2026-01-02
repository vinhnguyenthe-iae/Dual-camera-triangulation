"""MediaPipe Pose landmark connections.

Index reference (33 landmarks):
0: nose
1-6: eyes region
7-8: ears
9-10: mouth
11: left shoulder, 12: right shoulder
13: left elbow, 14: right elbow
15: left wrist, 16: right wrist
17: left pinky, 18: right pinky
19: left index, 20: right index
21: left thumb, 22: right thumb
23: left hip, 24: right hip
25: left knee, 26: right knee
27: left ankle, 28: right ankle
29: left heel, 30: right heel
31: left foot index, 32: right foot index
"""

BODY_CONNECTIONS = [
    # Shoulders and torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32),
    # Simple face outline
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (7, 8), (9, 10),
]

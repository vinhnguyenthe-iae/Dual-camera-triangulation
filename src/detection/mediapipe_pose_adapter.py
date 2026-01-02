import cv2
import os
import mediapipe as mp


class MediaPipePoseAdapter:
    """Unified adapter for MediaPipe Pose.

    - For mediapipe < 0.10: uses legacy solutions API.
    - For mediapipe >= 0.10: uses Tasks API (requires model asset).
    """

    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_asset_path: str = 'data/models/pose_landmarker_full.task'):
        self._mode = None  # 'solutions' or 'tasks'
        self._detector = None
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._model_asset_path = model_asset_path

        # Try legacy solutions API first
        try:
            from mediapipe import solutions as mp_solutions
            mp_pose = mp_solutions.pose
            self._detector = mp_pose.Pose(static_image_mode=False,
                                          model_complexity=1,
                                          enable_segmentation=False,
                                          min_detection_confidence=min_detection_confidence,
                                          min_tracking_confidence=min_tracking_confidence)
            self._mode = 'solutions'
            return
        except Exception:
            pass

        # Fallback to Tasks API
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(self._model_asset_path):
                raise FileNotFoundError(
                    f"MediaPipe Tasks model not found at {self._model_asset_path}. "
                    f"Download 'pose_landmarker_full.task' and place it there."
                )

            base_options = mp_python.BaseOptions(model_asset_path=self._model_asset_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._detector = mp_vision.PoseLandmarker.create_from_options(options)
            self._vision_module = mp_vision
            self._mode = 'tasks'
        except Exception as e:
            raise ImportError(
                "Failed to initialize MediaPipe Pose. Either install a version with 'mediapipe.solutions' "
                "or provide the Tasks model at models/pose_landmarker_full.task."
            ) from e

    def get_first_body_keypoints_px(self, frame):
        """Detect the first body and return pixel coordinates Nx2.

        Returns None if no pose detected.
        """
        h, w = frame.shape[:2]
        if self._mode == 'solutions':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)
            if results.pose_landmarks:
                pts = []
                for lm in results.pose_landmarks.landmark:
                    pts.append((lm.x * w, lm.y * h))
                return pts
            return None
        elif self._mode == 'tasks':
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._detector.detect(mp_image)
            if result.pose_landmarks:
                # Take first pose's landmarks
                pose = result.pose_landmarks[0]
                pts = []
                for lm in pose:
                    pts.append((lm.x * w, lm.y * h))
                return pts
            return None
        else:
            return None

    def close(self):
        if self._mode == 'solutions' and self._detector is not None:
            self._detector.close()

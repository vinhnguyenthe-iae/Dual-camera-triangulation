import cv2
import os
import mediapipe as mp


class MediaPipeAdapter:
    """Unified adapter for MediaPipe Hands.

    - For mediapipe < 0.10: uses legacy solutions API.
    - For mediapipe >= 0.10: uses Tasks API (requires model asset).
    """

    def __init__(self,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_asset_path: str = 'data/models/hand_landmarker.task'):
        self._mode = None  # 'solutions' or 'tasks'
        self._detector = None
        self._max_num_hands = max_num_hands
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._model_asset_path = model_asset_path

        # Try legacy solutions API first
        try:
            from mediapipe import solutions as mp_solutions
            mp_hands = mp_solutions.hands
            self._detector = mp_hands.Hands(static_image_mode=False,
                                            max_num_hands=max_num_hands,
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
                    f"Download 'hand_landmarker.task' and place it there."
                )

            base_options = mp_python.BaseOptions(model_asset_path=self._model_asset_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)
            self._vision_module = mp_vision
            self._mode = 'tasks'
        except Exception as e:
            raise ImportError(
                "Failed to initialize MediaPipe. Either install a version with 'mediapipe.solutions' "
                "or provide the Tasks model at data/models/hand_landmarker.task."
            ) from e

    def get_first_hand_keypoints_px(self, frame):
        """Detect the first hand and return pixel coordinates Nx2.

        Returns None if no hand detected.
        """
        h, w = frame.shape[:2]
        if self._mode == 'solutions':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                pts = []
                for lm in hand.landmark:
                    pts.append((lm.x * w, lm.y * h))
                return pts
            return None
        elif self._mode == 'tasks':
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._detector.detect(mp_image)
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                pts = []
                for lm in hand:
                    pts.append((lm.x * w, lm.y * h))
                return pts
            return None
        else:
            return None

    def close(self):
        if self._mode == 'solutions' and self._detector is not None:
            self._detector.close()
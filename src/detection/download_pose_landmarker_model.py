import os
import urllib.request


# Full model for higher accuracy; other variants: lite/heavy
# Latest model URL pattern from the mediapipe-models bucket
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
DEST_PATH = os.path.join("data", "models", "pose_landmarker_full.task")


def main():
    os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
    print(f"Downloading MediaPipe pose model from:\n{MODEL_URL}\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, DEST_PATH)
    except Exception as e:
        print("Download failed:", e)
        print("If the URL is blocked, manually download 'pose_landmarker_full.task' and place it at:")
        print(DEST_PATH)
        return
    print(f"Saved model to {DEST_PATH}")


if __name__ == "__main__":
    main()

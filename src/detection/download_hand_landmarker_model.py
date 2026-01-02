import os
import urllib.request


MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
DEST_PATH = os.path.join("data", "models", "hand_landmarker.task")


def main():
    os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
    print(f"Downloading MediaPipe hand model from:\n{MODEL_URL}\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, DEST_PATH)
    except Exception as e:
        print("Download failed:", e)
        print("If the URL is blocked, manually download 'hand_landmarker.task' from the MediaPipe repo/releases and place it at:")
        print(DEST_PATH)
        return
    print(f"Saved model to {DEST_PATH}")


if __name__ == "__main__":
    main()

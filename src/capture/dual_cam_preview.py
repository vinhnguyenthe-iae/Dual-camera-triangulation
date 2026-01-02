import sys
import time
import cv2


def open_cap(index):
    # Prefer DirectShow for virtual cams like iVCam; fallback to MSMF
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cap


def stack_side_by_side(f1, f2):
    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    h = max(h1, h2)
    f1_resized = cv2.resize(f1, (w1, h)) if h1 != h else f1
    f2_resized = cv2.resize(f2, (w2, h)) if h2 != h else f2
    return cv2.hconcat([f1_resized, f2_resized])


def main():
    cam1 = 1
    cam2 = 3
    if len(sys.argv) >= 3:
        try:
            cam1 = int(sys.argv[1])
            cam2 = int(sys.argv[2])
        except ValueError:
            print("Usage: python src/capture/dual_cam_preview.py <index1> <index2>")
            sys.exit(1)

    cap1 = open_cap(cam1)
    cap2 = open_cap(cam2)

    if not cap1.isOpened():
        print(f"Failed to open camera {cam1}")
        sys.exit(1)
    if not cap2.isOpened():
        print(f"Failed to open camera {cam2}")
        sys.exit(1)

    # Try setting a common resolution
    for cap in (cap1, cap2):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = f"Dual Preview ({cam1} & {cam2})"
    t_prev = time.time()
    frames = 0
    fps = 0.0
    while True:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 or not ok2:
            print("Failed to read from one of the cameras.")
            break

        frames += 1
        t_now = time.time()
        if t_now - t_prev >= 1.0:
            fps = frames / (t_now - t_prev)
            t_prev = t_now
            frames = 0

        view = stack_side_by_side(f1, f2)
        cv2.putText(view, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(window, view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

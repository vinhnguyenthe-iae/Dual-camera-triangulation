import sys
from pygrabber.dshow_graph import FilterGraph


def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for idx, name in enumerate(devices):
        print(f"{idx}: {name}")
    return devices


if __name__ == "__main__":
    devices = list_cameras()
    if not devices:
        print("No cameras found.")
        sys.exit(1)
    print("\nSelect by index with OpenCV, e.g.: python src/capture/ivcam_preview.py")

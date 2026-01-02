import sys
import numpy as np
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from .body_skeleton import BODY_CONNECTIONS


class PGBodyViewer:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Body Triangulation')
        self.view.setCameraPosition(distance=500)
        self.view.show()

        # Axes/grid for reference
        grid = gl.GLGridItem()
        grid.scale(100, 100, 1)
        self.view.addItem(grid)

        # Camera 3 preview window (2D)
        self.cam2_win = pg.GraphicsLayoutWidget(title='Camera 3 Preview')
        self.cam2_viewbox = self.cam2_win.addViewBox()
        self.cam2_viewbox.setAspectLocked(True)
        self.cam2_img = pg.ImageItem()
        self.cam2_viewbox.addItem(self.cam2_img)
        self.cam2_win.show()

        # Camera 0 preview window (2D)
        self.cam1_win = pg.GraphicsLayoutWidget(title='Camera 0 Preview')
        self.cam1_viewbox = self.cam1_win.addViewBox()
        self.cam1_viewbox.setAspectLocked(True)
        self.cam1_img = pg.ImageItem()
        self.cam1_viewbox.addItem(self.cam1_img)
        self.cam1_win.show()

        # Points
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=5, color=(0.3, 0.6, 1.0, 1.0))
        self.view.addItem(self.scatter)

        # Lines per connection
        self.line_items = []
        for _ in BODY_CONNECTIONS:
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.8, 0.8, 0.8, 1.0), width=2, antialias=True, mode='line_strip')
            self.view.addItem(line)
            self.line_items.append(line)

        # ArUco marker origin and axes
        self.marker_origin = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=8, color=(1.0, 1.0, 0.2, 1.0))
        self.view.addItem(self.marker_origin)
        self.axis_x = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.2, 0.2, 1.0), width=3, antialias=True, mode='line_strip')
        self.axis_y = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.2, 1.0, 0.2, 1.0), width=3, antialias=True, mode='line_strip')
        self.axis_z = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.2, 0.2, 1.0, 1.0), width=3, antialias=True, mode='line_strip')
        self.view.addItem(self.axis_x)
        self.view.addItem(self.axis_y)
        self.view.addItem(self.axis_z)

        # ArUco marker square outline (closed loop of 4 corners)
        self.marker_square = gl.GLLinePlotItem(pos=np.zeros((5, 3)), color=(1.0, 1.0, 0.0, 1.0), width=2, antialias=True, mode='line_strip')
        self.view.addItem(self.marker_square)

    def update(self, points_3d: np.ndarray):
        if points_3d is None or len(points_3d) == 0:
            return

        # Update scatter points
        self.scatter.setData(pos=points_3d)

        # Update lines per connection
        for (edge_idx, (i, j)) in enumerate(BODY_CONNECTIONS):
            if i < len(points_3d) and j < len(points_3d):
                seg = np.vstack([points_3d[i], points_3d[j]])
                self.line_items[edge_idx].setData(pos=seg)

    def process_events(self):
        QtWidgets.QApplication.processEvents()

    def update_marker(self, origin: np.ndarray, axes_endpoints: dict):
        """Update ArUco marker origin and axes.

        origin: shape (3,) in world coords
        axes_endpoints: dict with keys 'x','y','z' mapping to shape (3,) endpoints
        """
        if origin is not None:
            self.marker_origin.setData(pos=np.array([origin]))
        if axes_endpoints:
            ox = np.vstack([origin, axes_endpoints.get('x', origin)])
            oy = np.vstack([origin, axes_endpoints.get('y', origin)])
            oz = np.vstack([origin, axes_endpoints.get('z', origin)])
            self.axis_x.setData(pos=ox)
            self.axis_y.setData(pos=oy)
            self.axis_z.setData(pos=oz)

    def update_marker_square(self, corners_world: np.ndarray):
        """Update square outline using 4 marker corners in world coords (4x3)."""
        if corners_world is None or len(corners_world) != 4:
            return
        loop = np.vstack([corners_world, corners_world[0]])
        self.marker_square.setData(pos=loop)

    def update_cam2(self, frame_bgr):
        """Update camera 3 preview with a BGR frame."""
        if frame_bgr is None:
            return
        img = frame_bgr[:, :, ::-1]  # BGR -> RGB
        self.cam2_img.setImage(img, autoLevels=False)

    def update_cam1(self, frame_bgr):
        """Update camera 0 preview with a BGR frame."""
        if frame_bgr is None:
            return
        img = frame_bgr[:, :, ::-1]  # BGR -> RGB
        self.cam1_img.setImage(img, autoLevels=False)

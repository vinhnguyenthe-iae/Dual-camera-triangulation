import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

class Viewer:
    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)
        self.window = pg.GraphicsLayoutWidget(show=True, title="3D Hand Triangulation Viewer")
        self.window.setWindowTitle('3D Hand Triangulation Viewer')
        self.view = self.window.addViewBox()
        self.view.setCameraPosition(distance=2)
        self.view.setBackgroundColor('k')

        self.points_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), brush=pg.mkBrush('r'))
        self.view.addItem(self.points_item)

        self.lines_item = pg.PlotDataItem(pen=pg.mkPen('g', width=2))
        self.view.addItem(self.lines_item)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update(self):
        # This method should be connected to the data source to update the visualization
        pass

    def update_points(self, points):
        """Update the displayed points in the viewer."""
        if points is not None:
            self.points_item.setData(pos=points)

    def update_lines(self, lines):
        """Update the displayed lines in the viewer."""
        if lines is not None:
            self.lines_item.setData(lines)

    def run(self):
        QtGui.QApplication.instance().exec_()
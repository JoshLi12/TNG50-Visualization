import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor

from helper import load_galaxy_data
import os

bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TNG50 Galaxy Viewer")
        self.setMinimumSize(QSize(1200, 800))

        # Main layout
        widget = QWidget()
        layout = QVBoxLayout()

        # PyVista plot widget
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Button layout
        btn_layout = QHBoxLayout()
        self.origin_btn = QPushButton("Stellar Origin")
        self.velocity_btn = QPushButton("Velocity")
        self.metallicity_btn = QPushButton("Metallicity")

        for btn in [self.origin_btn, self.velocity_btn, self.metallicity_btn]:
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Load galaxy data once
        self.base_path = dest
        self.subfind_id = 333426
        self.data = load_galaxy_data(self.base_path, self.subfind_id)

        # Connect buttons
        self.origin_btn.clicked.connect(self.display_origin)
        self.velocity_btn.clicked.connect(self.display_velocity)
        self.metallicity_btn.clicked.connect(self.display_metallicity)

        # Initial render
        self.display_origin()

    def display_origin(self):
        self.plotter.clear()
        self.plotter.add_points(
            self.data['coords'], scalars=self.data['origin_tags'],
            cmap='coolwarm', render_points_as_spheres=True,
            point_size=2.5
        )
        self.plotter.add_scalar_bar(title="Stellar Origin")
        self.plotter.reset_camera()

    def display_velocity(self):
        self.plotter.clear()
        self.plotter.add_points(
            self.data['coords'], scalars=self.data['velocity_magnitude'],
            cmap='viridis', render_points_as_spheres=True,
            point_size=2.5
        )
        self.plotter.add_scalar_bar(title="Velocity (km/s)")
        self.plotter.reset_camera()

    def display_metallicity(self):
        self.plotter.clear()
        self.plotter.add_points(
            self.data['coords'], scalars=self.data['met'],
            cmap='inferno', render_points_as_spheres=True,
            point_size=2.5
        )
        self.plotter.add_scalar_bar(title="[Z/Z☉] (log scale)")
        self.plotter.reset_camera()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

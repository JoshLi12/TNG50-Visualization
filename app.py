import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor

from helper import load_galaxy_data
import os

import numpy as np


bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Galaxy Taskbar GUI")
        self.setMinimumSize(QSize(1200, 800))

        # Central widget with horizontal layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.plotter = QtInteractor(self)


        # Sidebar (vertical taskbar on the left)
        sidebar = QVBoxLayout()
        sidebar.setSpacing(1)  # Optional: spacing between buttons
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
    
        self.origin_btn = QPushButton("Stellar Origin")
        self.velocity_btn = QPushButton("Velocity")
        self.metallicity_btn = QPushButton("Metallicity")

        for btn in [self.origin_btn, self.velocity_btn, self.metallicity_btn]:
            sidebar.addWidget(btn)
        
        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.plotter.interactor)
        

        self.setCentralWidget(main_widget)

        # Load galaxy data once
        self.base_path = dest
        self.subfind_id = 333426
        self.data = load_galaxy_data(self.base_path, self.subfind_id)

        # Connect buttons
        self.origin_btn.clicked.connect(self.display_origin)
        self.velocity_btn.clicked.connect(self.display_velocity)
        self.metallicity_btn.clicked.connect(self.display_metallicity)

        # Plotter settings
        self.plotter.set_background([0.01, 0.01, 0.05])
        self.plotter.camera.zoom(10)  # >1 zooms in, <1 zooms out
        self.plotter.view_xy()
        self.plotter.enable_trackball_style()



        # Initial render
        self.display_origin()

        print(self.data['origin_tags'])

    def display_origin(self):
        self.plotter.clear()
        tag_colors = {
            1: [1.0, 0.2, 0.2, 0.6],  # Red (Main Progenitor)
            2: [0.2, 0.8, 0.2, 0.8],  # Green (FoF)
            3: [0.4, 0.4, 1.0, 0.8],  # Blue (External)
        }

        tags = self.data['origin_tags'].astype(int)
        rgba_colors = np.array([tag_colors.get(tag, [0, 0, 0, 0]) for tag in tags], dtype='f4')
        
        self.plotter.add_points(
            self.data['coords'],
            scalars=rgba_colors,
            rgba=True,
            render_points_as_spheres=True,
            point_size=2,
            show_scalar_bar=False
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

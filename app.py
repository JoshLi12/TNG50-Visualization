import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QVBoxLayout
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

        self.setWindowTitle("Galaxy Taskbar GUI")
        self.setMinimumSize(QSize(1200, 800))

        # Central widget with horizontal layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Sidebar (vertical taskbar on the left)
        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)  # Optional: spacing between buttons
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)

        # Example buttons
        sidebar.addWidget(QPushButton("Load Galaxy"))
        sidebar.addWidget(QPushButton("Toggle View"))
        sidebar.addWidget(QPushButton("Color Map"))
        sidebar.addWidget(QPushButton("Exit"))

        main_layout.addWidget(sidebar_widget)

        self.setCentralWidget(main_widget)

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

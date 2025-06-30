import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor
import pyvista as pv

from PyQt5.QtCore import QTimer

from helper import load_galaxy_data, get_galaxy_vel, compute_rvel, get_galaxy_coords
from matplotlib.colors import LinearSegmentedColormap

import os

import numpy as np

from matplotlib.colors import LinearSegmentedColormap



bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        pv.global_theme.allow_empty_mesh = True

        self.setWindowTitle("Galaxy Taskbar GUI")
        self.setMinimumSize(QSize(1200, 800))

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(lambda: self._update_radial_velocity(cloud, velocities))

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
        self.progenitor_btn = QPushButton("Main Progenitor")
        self.fof_btn = QPushButton("Friends of Friends")
        self.external_btn = QPushButton("External")


        for btn in [self.origin_btn, self.velocity_btn, self.metallicity_btn, self.progenitor_btn, self.fof_btn, self.external_btn]:
            sidebar.addWidget(btn)
        
        for btn in [self.progenitor_btn, self.fof_btn, self.external_btn]:
            btn.setCheckable(True)
            btn.setChecked(True)  # Show all by default
            btn.clicked.connect(self.display_origin)
            sidebar.addWidget(btn)
        
        self.origin_btn.clicked.connect(self.select_all_origins)

        # Metallicity Map
        

                
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

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)  # `self.close` is built-in from QMainWindow
        sidebar.addWidget(self.exit_btn)

        # Initial render
        self.display_origin()

        print(self.data['origin_tags'])

    def select_all_origins(self):
        self.progenitor_btn.setChecked(True)
        self.fof_btn.setChecked(True)
        self.external_btn.setChecked(True)
        self.display_origin()

    def display_origin(self):
        self.plotter.clear()

        tag_colors = {
            1: [1.0, 0.2, 0.2, 0.6],  # Red (Main Progenitor)
            2: [0.2, 0.8, 0.2, 0.8],  # Green (FoF)
            3: [0.4, 0.4, 1.0, 0.8],  # Blue (External)
        }

        tags = self.data['origin_tags'].astype(int)

        visible_mask = np.zeros(len(tags), dtype=bool)
        rgba_colors = np.zeros((len(tags), 4), dtype='f4')

        if self.progenitor_btn.isChecked():
            visible_mask |= tags == 1
            rgba_colors[tags == 1] = tag_colors[1]

        if self.fof_btn.isChecked():
            visible_mask |= tags == 2
            rgba_colors[tags == 2] = tag_colors[2]

        if self.external_btn.isChecked():
            visible_mask |= tags == 3
            rgba_colors[tags == 3] = tag_colors[3]

        coords = self.data['coords'][visible_mask]
        rgba_colors = rgba_colors[visible_mask]

        if len(coords) > 0:
            self.plotter.add_points(
                coords,
                scalars=rgba_colors,
                rgba=True,
                render_points_as_spheres=True,
                point_size=2,
                show_scalar_bar=False
            )
        else:
            print("No origin types selected — nothing to display.")
        self.plotter.reset_camera()


    def display_velocity(self):
        self.plotter.clear()

        # Re-load coordinates and rotation matrix (from helper)
        coords, rot_matrix = get_galaxy_coords(
            base_path=self.base_path,
            subfind_id=self.subfind_id
        )

        velocities = get_galaxy_vel(
            dest=self.base_path,
            subfind_id=self.subfind_id,
            v0=rot_matrix
        ).astype('f4')

        cloud = pv.PolyData(coords)
        cloud['v_radial'] = np.zeros(len(coords))  # placeholder for live updating

        # Custom blue-white-red color map
        colors = ['#2c7bb6', 'white', '#d7191c']
        custom_cmap = LinearSegmentedColormap.from_list("radial_cmap", colors)

        self.plotter.add_points(
            cloud,
            scalars='v_radial',
            cmap=custom_cmap,
            render_points_as_spheres=True,
            point_size=2.0,
            show_scalar_bar=True
        )
        self.plotter.add_scalar_bar(title="Radial Velocity (km/s)")
        self.plotter.reset_camera()

        def delay(caller, event):
            self.update_timer.start(50)

        def on_camera_move(caller, event):
            view_vector = self.plotter.camera.direction
            v_rad = compute_rvel(velocities, view_vector)
            cloud['v_radial'] = v_rad
            self.plotter.update_scalars(v_rad, render=True)

            p = np.percentile(v_rad, [1, 50, 99])
            print("v_rad percentiles (1%, 50%, 99%):", p)

        self.plotter.renderer.GetActiveCamera().AddObserver("ModifiedEvent", on_camera_move)

    def display_metallicity(self):
        self.plotter.clear()
        cloud = pv.PolyData(self.data['coords'])
        cloud['logZ'] = self.data['met']  # use log metallicity values from helper.py

        colors = ['#2c7bb6', 'white', '#d7191c']  # blue–white–red
        custom_cmap = LinearSegmentedColormap.from_list("radial_cmap", colors)
        self.plotter.add_points(
            cloud,
            scalars='logZ',
            cmap=custom_cmap,
            render_points_as_spheres=True,
            point_size=2.0,
            show_scalar_bar=False
        )
        self.plotter.add_scalar_bar(title="[Z/Z☉] (log scale)")
        self.plotter.reset_camera()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

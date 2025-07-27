import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit, QLabel, QFrame
)
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor
import pyvista as pv

from PyQt5.QtCore import QTimer

from helper import load_galaxy_data, get_galaxy_vel, compute_rvel, get_galaxy_coords
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

import os

import numpy as np

from matplotlib.colors import LinearSegmentedColormap



bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        pv.global_theme.allow_empty_mesh = True

        self.setWindowTitle("TNG50 Visualizer GUI")
        self.setMinimumSize(QSize(1200, 800))

        

        # Central widget with horizontal layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.plotter = QtInteractor(self)


        # Sidebar (vertical taskbar on the left)
        # sidebar = QVBoxLayout()
        # sidebar.setSpacing(1)  # Optional: spacing between buttons
        # sidebar_widget = QWidget()
        # sidebar_widget.setLayout(sidebar)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)
        
        sidebar_layout.addSpacing(30)

    
        self.origin_btn = QPushButton("Stellar Origin")
        self.velocity_btn = QPushButton("Velocity")
        self.metallicity_btn = QPushButton("Metallicity")
        self.progenitor_btn = QPushButton("Main Progenitor")
        self.fof_btn = QPushButton("Friends of Friends")
        self.external_btn = QPushButton("External")


        # Galaxy input section
        input_section = QVBoxLayout()

        # sidebar_widget.setFixedWidth(220)
        input_label = QLabel("Load Galaxy ID")
        input_section.addWidget(input_label)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("e.g. 333426")
        input_section.addWidget(self.input_box)


        # self.load_galaxy_button = QPushButton("Load Galaxy")
        # self.load_galaxy_button.clicked.connect(self.load_new_galaxy)
        self.input_box.returnPressed.connect(self.load_new_galaxy)


        
        input_section.addWidget(self.input_box)
        # input_section.addWidget(self.load_galaxy_button)
        input_section.addSpacing(10)
        view_section = QVBoxLayout()
        view_label = QLabel("View Modes")
        view_section.addWidget(view_label)
        view_section.addWidget(self.origin_btn)
        view_section.addWidget(self.velocity_btn)
        view_section.addWidget(self.metallicity_btn)

        view_section.addSpacing(30)

        tag_label = QLabel("Tag Filters")
        tag_section = QVBoxLayout()



        # for btn in [self.origin_btn, self.velocity_btn, self.metallicity_btn, self.progenitor_btn, self.fof_btn, self.external_btn]:
        #     sidebar.addWidget(btn)
        tag_section.addWidget(tag_label)
        for btn in [self.progenitor_btn, self.fof_btn, self.external_btn]:
            btn.setCheckable(True)
            btn.setChecked(True)  # Show all by default
            btn.clicked.connect(self.progenitor_click)
            tag_section.addWidget(btn)
        tag_section.addSpacing(30)



        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)  # `self.close` is built-in from QMainWindow
        # sidebar.addWidget(self.exit_btn)
        
        exit_section = QVBoxLayout()
        # exit_section.addSpacing(10)
        exit_section.addWidget(self.exit_btn)

        for section in [input_section, view_section, tag_section, exit_section]:
            group = QFrame()
            group.setLayout(section)
            sidebar_layout.addWidget(group)

        # Metallicity Map
        sidebar_container = QWidget()
        sidebar_container.setLayout(sidebar_layout)
        sidebar_container.setFixedWidth(250)

        main_layout.addWidget(sidebar_container)
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
        self.plotter.camera.zoom(15)  # >1 zooms in, <1 zooms out
        self.plotter.view_xy()
        self.plotter.enable_trackball_style()
        self.angle_text = self.plotter.add_text("View: initializing...", position='upper_left', font_size=10, color='purple')

        self.origin_btn.clicked.connect(self.origin_clicked)
        self.velocity_btn.clicked.connect(self.velocity_clicked)
        self.metallicity_btn.clicked.connect(self.met_clicked)

        self.plotter.add_axes(interactive=False)



        # Initial render
        self.display_origin()

        self.current_map = 1

        
    def load_new_galaxy(self):
        self.plotter.clear()
        text = self.input_box.text()
        if not text.isdigit():
            print("Invalid Subfind ID.")
            return

        self.subfind_id = int(text)
        print(f"Loading new galaxy: {self.subfind_id}")        

        self.data = load_galaxy_data(self.base_path, self.subfind_id)
        self.display_origin()

    def progenitor_click(self):
        if self.current_map == 1:
            self.display_origin()
        elif self.current_map == 2:
            self.display_velocity()
        elif self.current_map == 3:
            self.display_metallicity()


    def activate_all_origins(self):
        self.progenitor_btn.setChecked(True)
        self.fof_btn.setChecked(True)
        self.external_btn.setChecked(True)
    
    def origin_clicked(self):
        self.progenitor_btn.setChecked(True)
        self.fof_btn.setChecked(True)
        self.external_btn.setChecked(True)
        self.display_origin()

    def display_origin(self):
        self.plotter.reset_camera()
        self.plotter.clear()

        self.current_map = 1
        # self.activate_all_origins()
        tag_colors = {
            1: [1.0, 0.2, 0.2, 0.6],  # Red (Main Progenitor)
            2: [0.2, 0.8, 0.2, 0.8],  # Green (FoF)
            3: [0.4, 0.4, 1.0, 0.8],  # Blue (External)
        }

        coords_all = self.data['coords']
        tags_all = self.data['origin_tags'].astype(int)

        visible_mask = np.zeros(len(tags_all), dtype=bool)
        if self.progenitor_btn.isChecked():
            visible_mask |= tags_all == 1
        if self.fof_btn.isChecked():
            visible_mask |= tags_all == 2
        if self.external_btn.isChecked():
            visible_mask |= tags_all == 3

        # Filter coords and velocities
        coords = coords_all[visible_mask]
        tags = tags_all[visible_mask]

        # Build RGBA array from tag_colors
        rgba_colors = np.array([tag_colors.get(t, [0, 0, 0, 0]) for t in tags], dtype='f4')
        cloud = pv.PolyData(coords)

        print(len(coords))

        if len(coords) > 0:
            self.plotter.add_points(
                cloud,
                scalars=rgba_colors,
                rgba=True,
                render_points_as_spheres=True,
                point_size=2,
                show_scalar_bar=False
            )
        else:
            print("No origin types selected — nothing to display.")
        self.plotter.reset_camera()

    def velocity_clicked(self):
        self.progenitor_btn.setChecked(True)
        self.fof_btn.setChecked(True)
        self.external_btn.setChecked(True)
        self.display_velocity()
    
    def met_clicked(self):
        self.progenitor_btn.setChecked(True)
        self.fof_btn.setChecked(True)
        self.external_btn.setChecked(True)
        self.display_metallicity()

    def display_velocity(self):
        self.current_map = 2
        self.plotter.clear()
        # self.activate_all_origins()
        if self.current_map != 2:
            return
        

        # Load full data
        coords_all = self.data['coords']
        velocities_all = self.data['velocity_magnitude']
        tags = self.data['origin_tags'].astype(int)


        # Build mask
        visible_mask = np.zeros(len(tags), dtype=bool)
        if self.progenitor_btn.isChecked():
            visible_mask |= tags == 1
        if self.fof_btn.isChecked():
            visible_mask |= tags == 2
        if self.external_btn.isChecked():
            visible_mask |= tags == 3

        # Filter coords and velocities
        coords = coords_all[visible_mask]
        velocities = velocities_all[visible_mask]
        print(len(coords))

        if len(coords) == 0:
            print("No origin types selected — nothing to display.")
            return

        # Create cloud
        view_vector = self.plotter.camera.direction
        v_rad = compute_rvel(velocities, view_vector)
        cloud = pv.PolyData(coords)
        # cloud['v_radial'] = np.zeros(len(coords), dtype='f4')
        cloud['v_radial'] = v_rad

        # Setup colormap
        colors = ['#2c7bb6', 'white', '#d7191c']
        custom_cmap = LinearSegmentedColormap.from_list("radial_cmap", colors)
        # custom_cmap = matplotlib.colormaps['coolwarm']

        vmin, vmax = np.percentile(v_rad, [5, 95])  # Exclude outliers
        
        

        self.plotter.add_points(
            cloud,
            scalars='v_radial',
            cmap=custom_cmap,
            render_points_as_spheres=True,
            point_size=2.0,
            show_scalar_bar=False,
            clim=[vmin, vmax]
        )
        self._velocity_cloud = cloud
        self._velocity_mesh = cloud  
        self._visible_velocities = velocities

        self.plotter.add_scalar_bar(title="Radial Velocity (km/s)", color='white')
        self.plotter.reset_camera()

        # Timer and update function

        def _update_velocity_colors():

            view_vector = self.plotter.camera.direction
            v_rad = compute_rvel(self._visible_velocities, view_vector)

            self._velocity_cloud['v_radial'] = v_rad
            self.plotter.update_scalars(v_rad, render=True, mesh=self._velocity_mesh)

            p = np.percentile(v_rad, [1, 50, 99])
            print("v_rad percentiles (1%, 50%, 99%):", p)
        
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(_update_velocity_colors)

        def on_camera_move(caller, event):
            self._update_timer.start(5)

        # self.plotter.renderer.GetActiveCamera().AddObserver("ModifiedEvent", on_camera_move)
        self._camera_callback_tag = self.plotter.renderer.GetActiveCamera().AddObserver("ModifiedEvent", on_camera_move)



    def display_metallicity(self):
        self.current_map = 3
        self.plotter.clear()
        coords_all = self.data['coords']
        # cloud = pv.PolyData(self.data['coords'])
        # cloud['logZ'] = self.data['met']  # use log metallicity values from helper.py
        met_all = self.data['met']

        tags = self.data['origin_tags'].astype(int)


        visible_mask = np.zeros(len(tags), dtype=bool)
        if self.progenitor_btn.isChecked():
            visible_mask |= tags == 1
        if self.fof_btn.isChecked():
            visible_mask |= tags == 2
        if self.external_btn.isChecked():
            visible_mask |= tags == 3
        
        coords = coords_all[visible_mask]
        met = met_all[visible_mask]

        if len(coords) == 0:
            print("No origin types selected — nothing to display.")
            return

        cloud = pv.PolyData(coords)
        cloud['logZ'] = met

        colors = ["#7719aa", 'white', "#41c623"]  # blue–white–red
        custom_cmap = LinearSegmentedColormap.from_list("radial_cmap", colors)
        self.plotter.add_points(
            cloud,
            scalars='logZ',
            cmap=custom_cmap,
            render_points_as_spheres=True,
            point_size=2.0,
            show_scalar_bar=False
        )
        self.plotter.add_scalar_bar(
            title='(Z/Z☉) (log scale)',
            color='white',
        )
        self.plotter.reset_camera()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
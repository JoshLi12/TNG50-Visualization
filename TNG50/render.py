import os
import sys
import numpy as np
print("0")
import pygame
print("1.5")
from pygame.locals import DOUBLEBUF, OPENGL
print("2")

import moderngl
print("3")

from helper import get_galaxy_coords, bp_data
print("4")

window_size = (1000, 800)
pygame.init()
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)


ctx = moderngl.create_context()

subfind_id = 333426
theta = 90
phi = 0
angle = 0

coords = get_galaxy_coords(
    base_path=bp_data,
    subfind_id=subfind_id,
    theta=theta,
    phi=phi,
    angle=angle
).astype('f4')

print(f"Loaded {len(coords)} stellar particles for SubfindID {subfind_id}.")

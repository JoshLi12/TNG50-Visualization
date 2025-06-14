import os
import sys
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import moderngl
from helper import get_galaxy_coords, bp_data

sys.path.insert(0, bp_data + r"\code\illustris_python")
from findtags import create_tags


window_size = (1000, 800)
pygame.init()
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)


ctx = moderngl.create_context()

# Setup for render
pygame.init()
window_size = (1500, 800)
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()
clock = pygame.time.Clock()


subfind_id = 333426
theta = 90
phi = 0
angle = 0

bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

coords = get_galaxy_coords(
    base_path=dest,
    subfind_id=subfind_id,
    theta=theta,
    phi=phi,
    angle=angle
).astype('f4')

print(f"Loaded {len(coords)} stellar particles for SubfindID {subfind_id}.")

a1, a2, a3 = create_tags(subfind_id, bp_data + '/output/')
colors = np.zeros((len(coords), 3), dtype='f4')
colors[a1] = [1, 0, 0]  # Main Progenitor Branch
colors[a2] = [0, 1, 0]  # Friends of Friends
colors[a3] = [0, 0, 1]  # External Branch

# --- Normalize and center ---
coords /= 100.0  # scale down for display
vbo = ctx.buffer(coords.tobytes())
cbo = ctx.buffer(colors.tobytes())

prog = ctx.program(
    vertex_shader="""
    #version 330
    in vec3 in_pos;
    in vec3 in_color;
    out vec3 v_color;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(in_pos, 1.0);
        gl_PointSize = 5.0;
        v_color = in_color;
    }
    """,
    fragment_shader="""
    #version 330
    in vec3 v_color;
    out vec4 f_color;
    void main() {
        f_color = vec4(v_color, 1.0);
    }
    """
)

vao = ctx.vertex_array(
    prog,
    [(vbo, '3f', 'in_pos'), (cbo, '3f', 'in_color')]
)
def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype='f4')

proj = perspective(np.radians(45), window_size[0]/window_size[1], 0.1, 100.0)
zoom = 0  # initial zoom


running = True
while running:
    ctx.clear(0.0, 0.0, 0.0)

    view = np.eye(4, dtype='f4')
    view = view @ np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype='f4')
    view[3, 2] = zoom
    prog['mvp'].write((proj @ view).T.tobytes())

    vao.render(moderngl.POINTS)
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            zoom += event.y * 0.5  # scroll up = zoom in, down = zoom out
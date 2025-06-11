from Particles_starter import get_galaxy_coords

coords = get_galaxy_coords(
    base_path="N:/TNG50",
    subfind_id=333426,
    theta=90,
    phi=0,
    angle=0
).astype('f4')


window_size = (1000, 800)
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()
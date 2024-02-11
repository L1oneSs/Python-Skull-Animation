from bresenham import bresenham
import numpy as np
from lightning import get_I

width = 800
height = 800
z_buffer = np.ones((width, height))

def draw_rectangle(image, coords):
    for i in range(4):
        x1, y1, _ = coords[i]
        x2, y2, _ = coords[(i + 1) % 4]
        points = bresenham(int(x1), int(y1), int(x2), int(y2))

        for x, y in points:
            if x >= 800:
                x -= 1
            if y >= 800:
                y -= 1
            image[-y, x] = [0, 0, 0]

def draw_triangle(image, coords):
    for i in range(3):

        x1, y1, _ = coords[i]
        x2, y2, _ = coords[(i + 1) % 3]
        points = bresenham(int(x1), int(y1), int(x2), int(y2))

        for x, y in points:
            if x >= 800:
                x -= 1
            if y >= 800:
                y -= 1
            image[-y, x] = [0, 0, 0]


# Барецентрические координаты
# (a, b, c) - барицентрические координаты
# XA = aBA + bCA
def get_barycentric_coords(p, p1, p2, p3):
    matrix_A = np.array([
        [p1[0] - p3[0], p2[0] - p3[0]],
        [p1[1] - p3[1], p2[1] - p3[1]]
    ])
    vector_b = np.array([p[0] - p3[0], p[1] - p3[1]])

    try:
        solution = np.linalg.solve(matrix_A, vector_b)
    except np.linalg.LinAlgError:
        return 0, 0, 0

    c = 1.0 - solution.sum()
    return solution[0], solution[1], c

# z = a * z0 + b * z1 + c * z2
def get_z_coord(a, b, c, z0, z1, z2):
    z = a * z0 + b * z1 + c * z2
    return z

def get_border(v0, v1, v2):
    left = int(min(v0[0], v1[0], v2[0]))
    right = int(max(v0[0], v1[0], v2[0])) + 1
    bottom = int(min(v0[1], v1[1], v2[1]))
    top = int(max(v0[1], v1[1], v2[1])) + 1
    return left, right, bottom, top

def draw_edge(image, v0, v1, v2, bright):
    left, right, bottom, top = get_border(v0, v1, v2)
    for x in range(left, right):
        for y in range(bottom, top):
            point = np.array([x, y, 0])
            a, b, c = get_barycentric_coords(point, v0, v1, v2)
            if check_barycentric_coords_in_triangle(a, b, c):
                z_coordinate = get_z_coord(a, b, c, v0[2], v1[2], v2[2])
                if zbuffer_change(x, y, z_coordinate):
                    image[-y, x, 0] = bright
                    image[-y, x, 1] = bright
                    image[-y, x, 2] = bright

def draw_lightning(image, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, w, h, bright,
                 alpha, ia, id, i_s, ka, kd, ks, light_source, camera_source):
    left, right, bottom, top = get_border(v0, v1, v2)
    for x in range(left, right):
        for y in range(bottom, top):
            point = np.array([x, y, 0])
            a, b, c = get_barycentric_coords(point, v0, v1, v2)
            if check_barycentric_coords_in_triangle(a, b, c):
                z_coordinate = get_z_coord(a, b, c, v0[2], v1[2], v2[2])
                if zbuffer_change(x, y, z_coordinate):
                    uv = get_u_v(a, b, c, vt0, vt1, vt2)

                    texture_coord_y = w - 1 - int(uv[1] * w)
                    texture_coord_x = int(uv[0] * h)

                    normal = get_normal(a, b, c, vn0, vn1, vn2)
                    normal = np.array(normal)

                    I = get_I(normal, alpha, ia, id, i_s, ka, kd, ks, light_source, point, camera_source)
                    I = check_I(I)
                    global_bright = bright * (I / 255)
                    if global_bright > 255:
                        global_bright = 255
                    image[-y, x, 0] = texture[texture_coord_y, texture_coord_x, 0] * global_bright
                    image[-y, x, 1] = texture[texture_coord_y, texture_coord_x, 1] * global_bright
                    image[-y, x, 2] = texture[texture_coord_y, texture_coord_x, 2] * global_bright

def get_u_v(a, b, c, vt0, vt1, vt2):
    return [a * vt0[0] + b * vt1[0] + c * vt2[0], a * vt0[1] + b * vt1[1] + c * vt2[1]]

def get_normal(a, b, c, vn0, vn1, vn2):
    return [a * vn0[0] + b * vn1[0] + c * vn2[0], a * vn0[1] + b * vn1[1] + c * vn2[1],
            a * vn0[2] + b * vn1[2] + c * vn2[2]]

def zbuffer_change(x, y, z):
    global z_buffer
    if z <= z_buffer[x, y]:
        z_buffer[x, y] = z
        return 1
    return 0

def check_barycentric_coords_in_triangle(a, b, c):
    if 0 <= a and 0 <= b and 0 <= c:
        return True
    else:
        return False


def draw_texture(image, v0, v1, v2, vt0, vt1, vt2, texture, w, h, bright):
    x_min, x_max, y_min, y_max = get_border(v0, v1, v2)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = np.array([x, y])
            a, b, c = get_barycentric_coords(point, v0, v1, v2)
            if check_barycentric_coords_in_triangle(a, b, c):
                z_coordinate = get_z_coord(a, b, c, v0[2], v1[2], v2[2])
                if zbuffer_change(x, y, z_coordinate):
                    uv = get_u_v(a, b, c, vt0, vt1, vt2)
                    texture_coord_y = w - int(uv[1] * w)
                    texture_coord_x = int(uv[0] * h)
                    image[-y, x, 0] = texture[texture_coord_y, texture_coord_x, 0] * bright
                    image[-y, x, 1] = texture[texture_coord_y, texture_coord_x, 1] * bright
                    image[-y, x, 2] = texture[texture_coord_y, texture_coord_x, 2] * bright


def check_I(I):
    if I < 0:
        return np.abs(I)
    else:
        return I
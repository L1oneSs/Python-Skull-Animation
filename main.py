import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from drawer import draw_triangle, draw_rectangle, draw_edge, draw_texture, draw_lightning
from back_face_culling import back_face_culling
from matplotlib.animation import PillowWriter, FuncAnimation

file_path = 'Skull.txt'

f = []
v = []
v_test = []
vn = []
vt = []
width = 800
height = 800

with open(file_path, 'r') as file:
    for line in file:
        if not line:
            continue
        # vn
        if line.startswith('vn '):
            data = line.split()
            vn.append([float(data[1]), float(data[2]), float(data[3])])
        # vt
        elif line.startswith('vt '):
            data = line.split()
            vt.append([float(data[1]), float(data[2])])
        # v
        elif line.startswith('v '):
            data = line.split()
            v.append([float(data[1]), float(data[2]), float(data[3])])
            # f
        elif line.startswith('f '):
            data = line.split()
            face_vertices = []
            for vertex_data in data[1:]:
                vertex_indices = [int(index) - 1 if index else None for index in vertex_data.split('/')]
                face_vertices.append(vertex_indices)
            f.append(face_vertices)

# <--Mo2w-->

# Components of matrix R
alpha_x = 5
alpha_y = 15
alpha_z = 10


def rad_to_deg(alpha):
    return (alpha * np.pi) / 180


A = np.array([[1, 0, 0, 0],
              [0, np.cos(rad_to_deg(alpha_x)), -np.sin(rad_to_deg(alpha_x)), 0],
              [0, np.sin(rad_to_deg(alpha_x)), np.cos(rad_to_deg(alpha_x)), 0],
              [0, 0, 0, 1]])

B = np.array([[np.cos(rad_to_deg(alpha_y)), 0, np.sin(rad_to_deg(alpha_y)), 0],
              [0, 1, 0, 0],
              [-np.sin(rad_to_deg(alpha_y)), 0, np.cos(rad_to_deg(alpha_y)), 0],
              [0, 0, 0, 1]])

C = np.array([[np.cos(rad_to_deg(alpha_z)), -np.sin(rad_to_deg(alpha_z)), 0, 0],
              [np.sin(rad_to_deg(alpha_z)), np.cos(rad_to_deg(alpha_z)), 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = A @ B @ C

# Components of shift matrix

vector_c = [-2, 3, -2]

T = np.array([[1, 0, 0, vector_c[0]],
              [0, 1, 0, vector_c[1]],
              [0, 0, 1, vector_c[2]],
              [0, 0, 0, 1]])

# Components of scale matrix

scale = 0.8

S = np.array([[scale, 0, 0, 0],
              [0, scale, 0, 0],
              [0, 0, scale, 0],
              [0, 0, 0, 1]])

Mo2w = R @ T @ S

# Перемножение (Вершины и Нормали)
for i in range(0, len(v)):
    v[i] = (Mo2w @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mo2w.T) @ np.hstack([vn[i], 1]))[:3]

# <--Mw2c-->
A = np.array([2, 2, 2])

B = np.array([-2, -2, 0])

Tc = np.array([[1, 0, 0, -A[0]],
               [0, 1, 0, -A[1]],
               [0, 0, 1, -A[2]],
               [0, 0, 0, 1]])
# Вектор BA
BA = A - B

# Длина вектора BA
length_BA = np.sqrt(BA[0] ** 2 + BA[1] ** 2 + BA[2] ** 2)

# Нормализованный вектор BA
BA_norm = BA / length_BA

# Гамма
gamma = BA_norm
# Бета
beta = np.array([0, 1, 0]) - gamma[1] * gamma
# Альфа
# c = a * b = [a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1]
alpha = np.cross(beta, gamma)

Rc = np.array([[alpha.T[0], beta.T[0], gamma.T[0], 0],
               [alpha.T[1], beta.T[1], gamma.T[1], 0],
               [alpha.T[2], beta.T[2], gamma.T[2], 0],
               [0, 0, 0, 1]])

Mw2c = Rc @ Tc

# Перемножение (Вершины и Нормали)
for i in range(0, len(v)):
    v[i] = (Mw2c @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mw2c.T) @ np.hstack([vn[i], 1]))[:3]

# <--Mproj-->
l = min(v[i][0] for i in range(len(v)))
r = max(v[i][0] for i in range(len(v)))
t = max(v[i][1] for i in range(len(v)))
b = min(v[i][1] for i in range(len(v)))
n = min(v[i][2] for i in range(len(v)))
f_p = max(v[i][2] for i in range(len(v)))

Mproj = np.array([[2 / (r - l), 0, 0, -((r + l) / (r - l))],
                  [0, 2 / (t - b), 0, -((t + b) / (t - b))],
                  [0, 0, -2 / (f_p - n), -((f_p + n) / (f_p - n))],
                  [0, 0, 0, 1]])

# Перемножение (Вершины и нормали)

for i in range(0, len(v)):
    v[i] = (Mproj @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mproj.T) @ np.hstack([vn[i], 1]))[:3]


# <--Mviewport-->
x = 0
y = 0
ox = x + width // 2
oy = y + height // 2

Tw = np.array([[1, 0, 0, ox],
               [0, 1, 0, oy],
               [0, 0, 1, 1],
               [0, 0, 0, 1]])

Sw = np.array([[width // 2, 0, 0, 0],
               [0, height // 2, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

Mviewport = Tw @ Sw


for i in range(0, len(v)):
    v[i] = (Mviewport @ np.hstack([v[i], 1]))[:3]

# Перемножение (Вершины и Нормали)
for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mviewport.T) @ np.hstack([vn[i], 1]))[:3]

# 1. Проволочная модель
background_color = np.array([255, 255, 255])
image_1 = np.full((width, height, 3), background_color)

for edge in f:
    if len(edge) == 3:
        vertices_to_draw = [edge[0][0], edge[1][0], edge[2][0]]
        coords = [v[vertices_to_draw[0]], v[vertices_to_draw[1]],
                  v[vertices_to_draw[2]]]
        draw_triangle(image_1, coords)
    elif len(edge) == 4:
        vertices_to_draw = [edge[0][0], edge[1][0], edge[2][0], edge[3][0]]
        coords = [v[vertices_to_draw[0]], v[vertices_to_draw[1]],
                  v[vertices_to_draw[2]], v[vertices_to_draw[3]]]
        draw_rectangle(image_1, coords)

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(image_1)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('img1.png')
plt.show()

# 2. Модель с гранями
background_color = np.array([255, 255, 255])
image_2 = np.full((height, width, 3), background_color, np.uint8)
P = np.array([2, 2, 2])
coeff = 1 / 4000
color_bright = 0.1
for i in range(0, len(f)):
    current_face = f[i]

    if len(current_face) == 3:
        v0 = v[current_face[0][0]]
        v1 = v[current_face[1][0]]
        v2 = v[current_face[2][0]]
        intency = back_face_culling(P, v0, v1, v2, coeff, color_bright)
        if intency != None:
            draw_edge(image_2, v0, v1, v2, intency)

    elif len(current_face) == 4:
        v0 = v[current_face[0][0]]
        v1 = v[current_face[1][0]]
        v2 = v[current_face[2][0]]
        v3 = v[current_face[3][0]]
        intency = back_face_culling(P, v0, v1, v2, coeff, color_bright)
        if intency != None:
            draw_edge(image_2, v0, v1, v2, intency)
            draw_edge(image_2, v2, v3, v0, intency)

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(image_2)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('img2.png')
plt.show()

# 3. Модель с текстурами
background_color = np.array([255, 255, 255])
image_3 = np.full((height, width, 3), background_color, np.uint8)
P = np.array([2, 2, 2])
coeff = 1 / 8000
color_bright = 0.5
texture = matplotlib.image.imread('Skull.png')
weight_t = texture.shape[0]
height_t = texture.shape[1]

for i in range(0, len(f)):
    current_face = f[i]

    if len(current_face) == 3:
        v0 = v[current_face[0][0]]
        v1 = v[current_face[1][0]]
        v2 = v[current_face[2][0]]

        vt0 = vt[current_face[0][1]]
        vt1 = vt[current_face[1][1]]
        vt2 = vt[current_face[2][1]]

        intency = back_face_culling(P, v0, v1, v2, coeff, color_bright)
        if intency != None:
            draw_texture(image_3, v0, v1, v2, vt0, vt1, vt2, texture, weight_t, height_t, intency)

    elif len(current_face) == 4:
        v0 = v[current_face[0][0]]
        v1 = v[current_face[1][0]]
        v2 = v[current_face[2][0]]
        v3 = v[current_face[3][0]]

        vt0 = vt[current_face[0][1]]
        vt1 = vt[current_face[1][1]]
        vt2 = vt[current_face[2][1]]
        vt3 = vt[current_face[3][1]]

        intency = back_face_culling(P, v0, v1, v2, coeff, color_bright)
        if intency != None:
            draw_texture(image_3, v0, v1, v2, vt0, vt1, vt2, texture, weight_t, height_t, intency)
            draw_texture(image_3, v2, v3, v0, vt2, vt3, vt0, texture, weight_t, height_t, intency)

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(image_3)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('img3.png')
plt.show()


# 4. Анимация. Освещение. Модель Фонга
coeff_1 = 1 / 7000
color_bright_1 = 0.4
def check_params():
    global ia, id, i_s, ka, kd, ks
    # ia
    if ia[0] == 10:
        ia[0] = 200
    if ia[1] == 10:
        ia[1] = 200
    if ia[2] == 10:
        ia[2] = 200
    # id
    if id[0] == 10:
        id[0] = 200
    if id[1] == 10:
        id[1] = 200
    if id[2] == 10:
        id[2] = 200
    # is
    if i_s[0] == 10:
        i_s[0] = 200
    if i_s[1] == 10:
        i_s[1] = 200
    if i_s[2] == 10:
        i_s[2] = 200

    # ka
    if ka[0] > 1:
        ka[0] = 0
    if ka[1] > 1:
        ka[1] = 0
    if ka[2] > 1:
        ka[2] = 0
    # kd
    if kd[0] > 1:
        kd[0] = 0
    if kd[1] > 1:
        kd[1] = 0
    if kd[2] > 1:
        kd[2] = 0
    # ks
    if ks[0] > 1:
        ks[0] = 0
    if ks[1] > 1:
        ks[1] = 0
    if ks[2] > 1:
        ks[2] = 0


def change_params():
    global ia, id, i_s, ka, kd, ks, alpha
    unit_vector = np.array([1, 1, 1])
    vector_k = np.array([0.05, 0.05, 0.05])
    ia -= unit_vector
    id -= unit_vector
    i_s -= unit_vector

    ka += vector_k
    kd += vector_k
    ks += vector_k

    alpha -= 0.01


background_color = np.array([255, 255, 255])
image_4 = np.full((height, width, 3), background_color)

# Фоновое
ia = np.array([30, 30, 30])
ka = np.array([0.7, 0.7, 0.7])

# Диффузорное
light_sourse = np.array([-3, 3, 5])
camera_source = np.array([2, 2, 2])
id = np.array([160, 140, 150])
kd = np.array([0.8, 1, 0.6])

# Зеркальное
i_s = np.array([200, 120, 170])
ks = np.array([0.2, 0.3, 0.7])
alpha = 1

fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)


def update(frame):
    global image_4, background_color, ia, id, i_s, ka, kd, ks, alpha, light_sourse, camera_source
    image_4 = np.full((height, width, 3), background_color)
    check_params()

    for i in range(0, len(f)):
        current_face = f[i]

        if len(current_face) == 3:
            v0 = v[current_face[0][0]]
            v1 = v[current_face[1][0]]
            v2 = v[current_face[2][0]]

            vt0 = vt[current_face[0][1]]
            vt1 = vt[current_face[1][1]]
            vt2 = vt[current_face[2][1]]

            vn0 = vn[current_face[0][2]]
            vn1 = vn[current_face[1][2]]
            vn2 = vn[current_face[2][2]]

            intency = back_face_culling(P, v0, v1, v2, coeff_1, color_bright_1)
            if intency != None:
                draw_lightning(image_4, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, weight_t, height_t, intency,
                               alpha, ia, id, i_s, ka, kd, ks, light_sourse, camera_source)

        elif len(current_face) == 4:
            v0 = v[current_face[0][0]]
            v1 = v[current_face[1][0]]
            v2 = v[current_face[2][0]]
            v3 = v[current_face[3][0]]

            vt0 = vt[current_face[0][1]]
            vt1 = vt[current_face[1][1]]
            vt2 = vt[current_face[2][1]]
            vt3 = vt[current_face[3][1]]

            vn0 = vn[current_face[0][2]]
            vn1 = vn[current_face[1][2]]
            vn2 = vn[current_face[2][2]]
            vn3 = vn[current_face[2][2]]

            intency = back_face_culling(P, v0, v1, v2, coeff_1, color_bright_1)
            if intency != None:
                draw_lightning(image_4, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, weight_t, height_t, intency,
                               alpha, ia, id, i_s, ka, kd, ks, light_sourse, camera_source)
                draw_lightning(image_4, v2, v3, v0, vt2, vt3, vt0, vn2, vn3, vn0, texture, weight_t, height_t, intency,
                               alpha, ia, id, i_s, ka, kd, ks, light_sourse, camera_source)

    change_params()

    ax.imshow(image_4)
    ax.axis('off')
    plt.draw()

    plt.show()

    print("Кадр сделан")


animation_frames = 100
ani = FuncAnimation(fig, update, frames=animation_frames, repeat=False)

# Сохранение анимации в формате GIF
writer = PillowWriter(fps=10)
ani.save("Skull.gif", writer=writer)

plt.show()

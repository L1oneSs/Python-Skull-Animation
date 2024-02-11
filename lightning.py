import numpy as np


def get_R(L, N):
    # Вычисление вектора отражения R
    R = 2 * np.dot(N, L) * N - L
    return R


def get_V(point, camera_source):
    return camera_source - point


def calculate_IS(R, V, ks, i_s, alpha):
    cos = np.dot(R, V) / (np.linalg.norm(R) * np.linalg.norm(V))
    return np.dot(ks * (np.sign(cos) * (np.abs(cos)) ** (alpha)), i_s)


def get_L(l_source, point):
    return l_source - point


def calculate_ID(id, kd, N, L):
    cos = np.dot(N, L) / (np.linalg.norm(L) * np.linalg.norm(N))
    return np.dot(id, kd) * cos


def calculate_IA(ia, ka):
    return np.dot(ia, ka)


def get_I(N, alpha, ia, id, i_s, ka, kd, ks, light_source, point, camera_source):
    IA = calculate_IA(ia, ka)
    L = get_L(light_source, point)
    ID = calculate_ID(id, kd, N, L)
    R = get_R(L, N)
    V = get_V(point, camera_source)
    IS = calculate_IS(R, V, ks, i_s, alpha)
    return IA + IS + ID


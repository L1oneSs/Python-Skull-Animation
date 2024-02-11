import numpy as np

def reorder_vertices(vertex_a, vertex_b, vertex_c):
    x_min_val = min(vertex_a[0], vertex_b[0], vertex_c[0])

    if vertex_a[0] == x_min_val:
        return vertex_c, vertex_a, vertex_b
    elif vertex_c[0] == x_min_val:
        return vertex_b, vertex_c, vertex_a
    else:
        return vertex_a, vertex_b, vertex_c

def back_face_culling(P, vertex_a, vertex_b, vertex_c, coeff, color_bright):
    # Ставим вершину с минимальным x как смежную
    point_a, point_b, point_c = reorder_vertices(vertex_a, vertex_b, vertex_c)

    normal_vec = np.cross((point_b - point_a), (point_b - point_c))

    # Меняем порядок обхода
    if np.any(normal_vec <= 0):
        point_a, point_b = point_b, point_a

    translation_vec = (vertex_a - P)
    normal_vec = np.cross((point_b - point_a), (point_c - point_b))

    result = np.dot(translation_vec, normal_vec)
    result = result * coeff + color_bright

    return check_result(result)

def check_result(result):
    if result is not None and result >= 0:
        result *= 255
        return 255 if result > 255 else result
    else:
        return None

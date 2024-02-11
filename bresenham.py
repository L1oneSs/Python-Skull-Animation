def bresenham(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx - dy

    points = []

    while x1 != x2 or y1 != y2:
        points.append((x1, y1))
        e2 = 2 * error
        # Следующий пиксель по горизонатли ближе к истинной линии
        if e2 > -dy:
            error -= dy
            x1 += sx
        # Следующий пиксель по вертикали ближе к истинной линии
        if e2 < dx:
            error += dx
            y1 += sy

    points.append((x2, y2))
    return points
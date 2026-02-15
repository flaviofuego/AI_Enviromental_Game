"""
Physics and vector math utilities used by both game and training.
Pure Python — no pygame dependency.
"""
import math


def calculate_vector(start, end):
    """Return the vector from *start* to *end* as a two-element list."""
    if hasattr(start, "x"):
        return [end.x - start.x, end.y - start.y]
    return [end[0] - start[0], end[1] - start[1]]


def vector_length(vector) -> float:
    """Return the length (magnitude) of a 2-element vector."""
    if hasattr(vector, "x"):
        return math.hypot(vector.x, vector.y)
    return math.hypot(vector[0], vector[1])


def normalize_vector(vector):
    """Return a unit-length version of *vector* (or [0, 0] if zero-length)."""
    length = vector_length(vector)
    if length > 0:
        if hasattr(vector, "x"):
            return [vector.x / length, vector.y / length]
        return [vector[0] / length, vector[1] / length]
    return [0.0, 0.0]


def dot_product(a, b) -> float:
    """Return the dot product of two 2-element vectors."""
    ax, ay = (a.x, a.y) if hasattr(a, "x") else (a[0], a[1])
    bx, by = (b.x, b.y) if hasattr(b, "x") else (b[0], b[1])
    return ax * bx + ay * by


def line_circle_intersection(line_start, line_end, circle_center, circle_radius):
    """
    Return intersection points between a line segment and a circle,
    or ``None`` if the segment does not intersect.
    """
    ls = [line_start[0], line_start[1]] if not isinstance(line_start, list) else line_start
    le = [line_end[0], line_end[1]] if not isinstance(line_end, list) else line_end
    cc = [circle_center[0], circle_center[1]] if not isinstance(circle_center, list) else circle_center

    line_vec = calculate_vector(ls, le)
    line_len = vector_length(line_vec)
    if line_len == 0:
        return None
    line_dir = normalize_vector(line_vec)

    start_to_center = calculate_vector(ls, cc)
    projection = start_to_center[0] * line_dir[0] + start_to_center[1] * line_dir[1]
    projection = max(0.0, min(projection, line_len))

    closest = [ls[0] + line_dir[0] * projection, ls[1] + line_dir[1] * projection]
    closest_to_center = calculate_vector(closest, cc)
    distance = vector_length(closest_to_center)

    if distance > circle_radius:
        return None

    offset = math.sqrt(max(0.0, circle_radius * circle_radius - distance * distance))
    intersection1 = [closest[0] - line_dir[0] * offset, closest[1] - line_dir[1] * offset]
    intersection2 = [closest[0] + line_dir[0] * offset, closest[1] + line_dir[1] * offset]
    return intersection1, intersection2

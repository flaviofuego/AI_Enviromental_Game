import math

def calculate_vector(start, end):
    """Calculate the vector from start point to end point"""
    if isinstance(start, Vector) and isinstance(end, Vector):
        return Vector(end.x - start.x, end.y - start.y)
    return Vector(end[0] - start[0], end[1] - start[1])

def normalize_vector(vector):
    """Normalize a vector to unit length"""
    if isinstance(vector, Vector):
        length = vector.length
        if length > 0:
            return Vector(vector.x / length, vector.y / length)
        return Vector()
    length = math.hypot(vector[0], vector[1])
    if length > 0:
        return [vector[0] / length, vector[1] / length]
    return [0, 0]

def vector_length(vector):
    """Calculate the length of a vector"""
    if isinstance(vector, Vector):
        return vector.length
    return math.hypot(vector[0], vector[1])

def line_circle_intersection(line_start, line_end, circle_center, circle_radius):
    """Calculate intersection points between a line and a circle"""
    # Convert Vector objects to lists if needed
    if isinstance(line_start, Vector):
        line_start = [line_start.x, line_start.y]
    if isinstance(line_end, Vector):
        line_end = [line_end.x, line_end.y]
    if isinstance(circle_center, Vector):
        circle_center = [circle_center.x, circle_center.y]
    
    # Vector from line start to end
    line_vec = calculate_vector(line_start, line_end)
    line_length = vector_length(line_vec)
    
    # Normalize line vector
    line_vec = normalize_vector(line_vec)
    
    # Vector from line start to circle center
    start_to_center = calculate_vector(line_start, circle_center)
    
    # Project circle center onto line
    if isinstance(line_vec, Vector):
        projection = start_to_center.x * line_vec.x + start_to_center.y * line_vec.y
    else:
        projection = start_to_center[0] * line_vec[0] + start_to_center[1] * line_vec[1]
    projection = max(0, min(projection, line_length))
    
    # Closest point on line to circle center
    if isinstance(line_vec, Vector):
        closest_point = [
            line_start[0] + line_vec.x * projection,
            line_start[1] + line_vec.y * projection
        ]
    else:
        closest_point = [
            line_start[0] + line_vec[0] * projection,
            line_start[1] + line_vec[1] * projection
        ]
    
    # Vector from closest point to circle center
    closest_to_center = calculate_vector(closest_point, circle_center)
    distance = vector_length(closest_to_center)
    
    # No intersection if distance is greater than radius
    if distance > circle_radius:
        return None
    
    # Calculate intersection points
    offset = math.sqrt(circle_radius * circle_radius - distance * distance)
    if isinstance(line_vec, Vector):
        intersection1 = [
            closest_point[0] - line_vec.x * offset,
            closest_point[1] - line_vec.y * offset
        ]
        intersection2 = [
            closest_point[0] + line_vec.x * offset,
            closest_point[1] + line_vec.y * offset
        ]
    else:
        intersection1 = [
            closest_point[0] - line_vec[0] * offset,
            closest_point[1] - line_vec[1] * offset
        ]
        intersection2 = [
            closest_point[0] + line_vec[0] * offset,
            closest_point[1] + line_vec[1] * offset
        ]
    
    return intersection1, intersection2

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x / scalar, self.y / scalar)
        return NotImplemented
    
    def dot(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        return NotImplemented

    @property
    def length(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        length = self.length
        if length > 0:
            return Vector(self.x / length, self.y / length)
        return Vector()

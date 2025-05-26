import math

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

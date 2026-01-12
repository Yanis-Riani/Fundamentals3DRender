from __future__ import annotations
import math


class Vector3():
    vx: float
    vy: float
    vz: float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.vx = x
        self.vy = y
        self.vz = z

    @property
    def x(self) -> float:
        return self.vx
    
    @x.setter
    def x(self, value: float):
        self.vx = value

    @property
    def y(self) -> float:
        return self.vy
    
    @y.setter
    def y(self, value: float):
        self.vy = value

    @property
    def z(self) -> float:
        return self.vz
    
    @z.setter
    def z(self, value: float):
        self.vz = value

    def __add__(self, v: Vector3) -> Vector3:
        vres = Vector3()
        vres.vx = self.vx + v.vx
        vres.vy = self.vy + v.vy
        vres.vz = self.vz + v.vz
        return vres

    def __sub__(self, v: Vector3) -> Vector3:
        vres = Vector3()
        vres.vx = self.vx - v.vx
        vres.vy = self.vy - v.vy
        vres.vz = self.vz - v.vz
        return vres

    def __mul__(self, v: Vector3 | float | int) -> float | Vector3:
        if isinstance(v, (float, int)):
            vres = Vector3()
            vres.vx = self.vx * v
            vres.vy = self.vy * v
            vres.vz = self.vz * v
            return vres
        return self.vx * v.vx + self.vy * v.vy + self.vz * v.vz

    def cross_product(self, v: Vector3) -> Vector3:
        """Calculates the cross product of two vectors."""
        vres = Vector3()
        vres.vx = self.vy * v.vz - self.vz * v.vy
        vres.vy = -(self.vx * v.vz - self.vz * v.vx)
        vres.vz = self.vx * v.vy - self.vy * v.vx
        return vres

    def dot_product(self, v: Vector3) -> float:
        """Calculates the dot product of two vectors."""
        return self.__mul__(v)

    def __neg__(self) -> Vector3:
        """Returns the negation of the vector."""
        return Vector3(-self.vx, -self.vy, -self.vz)

    def __rmul__(self, k: float) -> Vector3:
        vres = Vector3()
        vres.vx = self.vx * k
        vres.vy = self.vy * k
        vres.vz = self.vz * k
        return vres

    def norm(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def normalize(self) -> Vector3:
        norm_val = self.norm()
        if norm_val == 0:
            return Vector3()
        val = 1 / norm_val
        return self.__rmul__(val)

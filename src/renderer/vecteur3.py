from __future__ import annotations
import math


class Vecteur():
    vx: float
    vy: float
    vz: float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.vx = x
        self.vy = y
        self.vz = z

    def __add__(self, v: Vecteur) -> Vecteur:
        vres = Vecteur()
        vres.vx = self.vx + v.vx
        vres.vy = self.vy + v.vy
        vres.vz = self.vz + v.vz
        return vres

    def __sub__(self, v: Vecteur) -> Vecteur:
        vres = Vecteur()
        vres.vx = self.vx - v.vx
        vres.vy = self.vy - v.vy
        vres.vz = self.vz - v.vz
        return vres

    def __mul__(self, v: Vecteur) -> float:    # produit scalaire
        return self.vx * v.vx + self.vy * v.vy + self.vz * v.vz

    def dot(self, v: Vecteur) -> Vecteur:
        """This is the cross product"""
        vres = Vecteur()
        vres.vx = self.vy * v.vz - self.vz * v.vy
        vres.vy = -(self.vx * v.vz - self.vz * v.vx)
        vres.vz = self.vx * v.vy - self.vy * v.vx
        return vres

    def __neg__(self) -> Vecteur:
        """Returns the negation of the vector."""
        return Vecteur(-self.vx, -self.vy, -self.vz)

    def __rmul__(self, k: float) -> Vecteur:
        vres = Vecteur()
        vres.vx = self.vx * k
        vres.vy = self.vy * k
        vres.vz = self.vz * k
        return vres

    def norm(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def normer(self) -> Vecteur:
        norm_val = self.norm()
        if norm_val == 0:
            return Vecteur()
        val = 1 / norm_val
        return self.__rmul__(val)
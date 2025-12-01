from __future__ import annotations

from typing import List

from src.renderer import vecteur3


class Matrix4:
    def __init__(self):
        self.mat: List[List[float]] = [[0.0] * 4 for _ in range(4)]

    @staticmethod
    def identity() -> Matrix4:
        m = Matrix4()
        m.mat[0][0] = 1.0
        m.mat[1][1] = 1.0
        m.mat[2][2] = 1.0
        m.mat[3][3] = 1.0
        return m

    def __mul__(self, other: Matrix4) -> Matrix4:
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                sum_val = 0.0
                for k in range(4):
                    sum_val += self.mat[i][k] * other.mat[k][j]
                result.mat[i][j] = sum_val
        return result

    def transform_point(self, p: vecteur3.Vecteur) -> vecteur3.Vecteur:
        # Assumes p is a point, so w = 1
        x = p.vx * self.mat[0][0] + p.vy * self.mat[1][0] + p.vz * self.mat[2][0] + self.mat[3][0]
        y = p.vx * self.mat[0][1] + p.vy * self.mat[1][1] + p.vz * self.mat[2][1] + self.mat[3][1]
        z = p.vx * self.mat[0][2] + p.vy * self.mat[1][2] + p.vz * self.mat[2][2] + self.mat[3][2]
        w = p.vx * self.mat[0][3] + p.vy * self.mat[1][3] + p.vz * self.mat[2][3] + self.mat[3][3]

        if w != 0.0:
            x /= w
            y /= w
            z /= w

        return vecteur3.Vecteur(x, y, z)

    @staticmethod
    def look_at(eye: vecteur3.Vecteur, target: vecteur3.Vecteur, up: vecteur3.Vecteur) -> Matrix4:
        zaxis = (target - eye).normer()
        xaxis = (up.dot(zaxis)).normer()
        yaxis = zaxis.dot(xaxis)

        m = Matrix4.identity()
        
        m.mat[0][0] = xaxis.vx
        m.mat[1][0] = xaxis.vy
        m.mat[2][0] = xaxis.vz

        m.mat[0][1] = yaxis.vx
        m.mat[1][1] = yaxis.vy
        m.mat[2][1] = yaxis.vz

        m.mat[0][2] = -zaxis.vx
        m.mat[1][2] = -zaxis.vy
        m.mat[2][2] = -zaxis.vz

        m.mat[3][0] = -xaxis * eye
        m.mat[3][1] = -yaxis * eye
        m.mat[3][2] = -zaxis * eye
        
        return m

    @staticmethod
    def create_translation(t: vecteur3.Vecteur) -> Matrix4:
        m = Matrix4.identity()
        m.mat[3][0] = t.vx
        m.mat[3][1] = t.vy
        m.mat[3][2] = t.vz
        return m

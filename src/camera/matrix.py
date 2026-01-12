from __future__ import annotations

from typing import List

from ..renderer import vecteur3


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

    def transform_point(self, p: vecteur3.Vector3) -> vecteur3.Vector3:
        # Assumes p is a point, so w = 1
        x = p.x * self.mat[0][0] + p.y * self.mat[1][0] + p.z * self.mat[2][0] + self.mat[3][0]
        y = p.x * self.mat[0][1] + p.y * self.mat[1][1] + p.z * self.mat[2][1] + self.mat[3][1]
        z = p.x * self.mat[0][2] + p.y * self.mat[1][2] + p.z * self.mat[2][2] + self.mat[3][2]
        w = p.x * self.mat[0][3] + p.y * self.mat[1][3] + p.z * self.mat[2][3] + self.mat[3][3]

        if w != 0.0:
            x /= w
            y /= w
            z /= w

        return vecteur3.Vector3(x, y, z)

    def transpose(self) -> Matrix4:
        result = Matrix4()
        for i in range(3): # Only transpose the 3x3 rotation part
            for j in range(3):
                result.mat[i][j] = self.mat[j][i]
        
        # Preserve translation and w component
        result.mat[3][0] = self.mat[3][0]
        result.mat[3][1] = self.mat[3][1]
        result.mat[3][2] = self.mat[3][2]
        result.mat[3][3] = self.mat[3][3] 
        return result

    def inverse(self) -> Matrix4:
        # Assuming the matrix is an affine transformation matrix (rotation and translation)
        # M = [ R | t ]
        #     [ 0 | 1 ]
        # M_inv = [ R^T | -R^T * t ]
        #         [ 0   | 1        ]
        
        result = Matrix4()
        # Transpose the 3x3 rotation part (R^T)
        for i in range(3):
            for j in range(3):
                result.mat[i][j] = self.mat[j][i]
        
        # Calculate -R^T * t
        t_vec = vecteur3.Vector3(self.mat[3][0], self.mat[3][1], self.mat[3][2])
        
        # Apply R^T to -t
        inv_t_x = -(result.mat[0][0] * t_vec.x + result.mat[1][0] * t_vec.y + result.mat[2][0] * t_vec.z)
        inv_t_y = -(result.mat[0][1] * t_vec.x + result.mat[1][1] * t_vec.y + result.mat[2][1] * t_vec.z)
        inv_t_z = -(result.mat[0][2] * t_vec.x + result.mat[1][2] * t_vec.y + result.mat[2][2] * t_vec.z)

        result.mat[3][0] = inv_t_x
        result.mat[3][1] = inv_t_y
        result.mat[3][2] = inv_t_z
        result.mat[3][3] = 1.0 # The w component remains 1

        # Fill the bottom row with 0s except for the last element
        result.mat[0][3] = 0.0
        result.mat[1][3] = 0.0
        result.mat[2][3] = 0.0
        
        return result

    @staticmethod
    def look_at(eye: vecteur3.Vector3, target: vecteur3.Vector3, up: vecteur3.Vector3) -> Matrix4:
        zaxis = (target - eye).normalize() # Forward (+Z in camera space)
        xaxis = (up.cross_product(zaxis)).normalize() # Camera Right
        yaxis = zaxis.cross_product(xaxis).normalize() # Camera Up

        m = Matrix4.identity()
        
        # Columns represent where the basis vectors map to
        # In transform_point: p.x * Col0 + p.y * Col1 + p.z * Col2 + Col3
        
        m.mat[0][0] = xaxis.x
        m.mat[1][0] = xaxis.y
        m.mat[2][0] = xaxis.z

        m.mat[0][1] = yaxis.x
        m.mat[1][1] = yaxis.y
        m.mat[2][1] = yaxis.z

        m.mat[0][2] = zaxis.x
        m.mat[1][2] = zaxis.y
        m.mat[2][2] = zaxis.z

        m.mat[3][0] = -xaxis * eye
        m.mat[3][1] = -yaxis * eye
        m.mat[3][2] = -zaxis * eye
        
        return m

    @staticmethod
    def create_translation(t: vecteur3.Vector3) -> Matrix4:
        m = Matrix4.identity()
        m.mat[3][0] = t.x
        m.mat[3][1] = t.y
        m.mat[3][2] = t.z
        return m
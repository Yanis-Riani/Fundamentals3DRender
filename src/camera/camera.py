from __future__ import annotations

import math

from ..renderer import vecteur3
from . import matrix


class Camera:
    def __init__(self, distance: float = 10.0):
        self.distance = distance
        self.azimuth = math.pi  # Rotation around Y axis
        self.elevation = 0.0    # Rotation around X axis
        self.target = vecteur3.Vecteur(0.0, 0.0, 0.0)
        self.up = vecteur3.Vecteur(0.0, 1.0, 0.0)
        self.position = vecteur3.Vecteur(0.0, 0.0, self.distance)
        self.update_position()

    def update_position(self) -> None:
        # Spherical to Cartesian coordinates
        x = self.target.vx + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.target.vy + self.distance * math.sin(self.elevation)
        z = self.target.vz + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        self.position = vecteur3.Vecteur(x, y, z)

    def get_view_matrix(self) -> matrix.Matrix4:
        return matrix.Matrix4.look_at(self.position, self.target, self.up)

    def rotate(self, dx: float, dy: float) -> None:
        sensitivity = 0.02
        self.azimuth -= dx * sensitivity
        self.elevation -= dy * sensitivity
        self.update_position()

    def zoom(self, amount: float) -> None:
        """Adjusts the camera's distance from the target."""
        self.distance -= amount * 0.5 # Adjust sensitivity as needed
        self.distance = max(0.1, self.distance) # Prevent going too close
        self.update_position()

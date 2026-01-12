from __future__ import annotations

import math

from ..renderer import vecteur3
from . import matrix


class Camera:
    def __init__(self, distance: float = 400.0):
        self.distance = distance
        self.azimuth = math.pi / 4  # Top-Right perspective
        self.elevation = math.pi / 6    # Looking slightly down
        self.target = vecteur3.Vector3(0.0, 0.0, 0.0)
        self.world_up = vecteur3.Vector3(0.0, 1.0, 0.0)
        self.position = vecteur3.Vector3(0.0, 0.0, self.distance)
        self.update_position()

    def update_position(self) -> None:
        # Spherical to Cartesian coordinates
        # Position is relative to Target
        x = self.target.x + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.target.y + self.distance * math.sin(self.elevation)
        z = self.target.z + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        self.position = vecteur3.Vector3(x, y, z)

    def get_view_matrix(self) -> matrix.Matrix4:
        # Determine Up vector based on elevation to avoid flipping at poles
        normalized_elevation = self.elevation % (2 * math.pi)
        if normalized_elevation > math.pi:
            normalized_elevation -= 2 * math.pi

        up_vector = self.world_up
        if normalized_elevation < -math.pi/2 or normalized_elevation > math.pi/2:
            up_vector = vecteur3.Vector3(0.0, -1.0, 0.0)

        return matrix.Matrix4.look_at(self.position, self.target, up_vector)

    def rotate(self, dx: float, dy: float) -> None:
        sensitivity = 0.01
        self.azimuth -= dx * sensitivity
        self.elevation -= dy * sensitivity
        self.update_position()

    def zoom(self, amount: float) -> None:
        """Adjusts the camera's distance from the target."""
        # Scroll delta is usually 120. Scale it down.
        self.distance -= amount * 0.1 
        self.distance = max(1.0, self.distance) # Prevent going behind target or zero
        self.update_position()

    def pan(self, dx: float, dy: float) -> None:
        """Moves the camera target (and position) laterally."""
        # Calculate Forward, Right, Up vectors
        # Forward is Target - Position
        forward = (self.target - self.position).normalize()
        
        # Right is Cross(Forward, WorldUp)
        right = forward.cross_product(self.world_up).normalize()
        
        # Up_Cam is Cross(Right, Forward)
        up_cam = right.cross_product(forward).normalize()
        
        # Sensitivity scales with distance (further away = faster pan)
        sensitivity = self.distance * 0.001
        
        move_vec = (right * (-dx * sensitivity)) + (up_cam * (dy * sensitivity))
        
        self.target = self.target + move_vec
        self.update_position()
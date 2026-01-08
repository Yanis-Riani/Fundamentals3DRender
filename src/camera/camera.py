from __future__ import annotations

import math

from ..renderer import vecteur3
from . import matrix


class Camera:
    def __init__(self, distance: float = 400.0):
        self.distance = distance
        self.azimuth = math.pi  # Rotation around Y axis
        self.elevation = 0.0    # Rotation around X axis
        self.target = vecteur3.Vecteur(0.0, 0.0, 0.0)
        self.world_up = vecteur3.Vecteur(0.0, 1.0, 0.0)
        self.position = vecteur3.Vecteur(0.0, 0.0, self.distance)
        self.update_position()

    def update_position(self) -> None:
        # Spherical to Cartesian coordinates
        # Position is relative to Target
        x = self.target.vx + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.target.vy + self.distance * math.sin(self.elevation)
        z = self.target.vz + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        self.position = vecteur3.Vecteur(x, y, z)

    def get_view_matrix(self) -> matrix.Matrix4:
        # Determine Up vector based on elevation to avoid flipping at poles
        normalized_elevation = self.elevation % (2 * math.pi)
        if normalized_elevation > math.pi:
            normalized_elevation -= 2 * math.pi

        up_vector = self.world_up
        if normalized_elevation < -math.pi/2 or normalized_elevation > math.pi/2:
            up_vector = vecteur3.Vecteur(0.0, -1.0, 0.0)

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
        forward = (self.target - self.position).normer()
        
        # Right is Cross(Forward, WorldUp)
        # Note: vecteur3.dot is cross product alias
        right = forward.produitVectoriel(self.world_up).normer()
        
        # Up_Cam is Cross(Right, Forward)
        up_cam = right.produitVectoriel(forward).normer()
        
        # Sensitivity scales with distance (further away = faster pan)
        sensitivity = self.distance * 0.001
        
        # Move target
        # -dx moves Right (screen space logic: drag left = move camera right? No, drag left moves view left)
        # Blender: Shift+Middle Drag Right -> View moves Right (Target moves Left relative to camera? No, Target moves Right)
        # Let's try direct mapping first.
        # -dx because typically dragging "world" means moving camera opposite?
        # Actually usually Pan follows mouse. Mouse Right -> Target Right.
        
        move_vec = (right * (-dx * sensitivity)) + (up_cam * (dy * sensitivity))
        
        self.target = self.target + move_vec
        self.update_position()
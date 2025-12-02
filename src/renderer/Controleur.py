import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")

import tkinter.filedialog
from typing import Any, Callable, List, Tuple

from ..camera import camera, matrix
from . import Import_scene, Modele, vecteur3

# Type aliases for clarity (re-defined or imported from Modele for consistency)
Point2D = Tuple[int, int]
Point3D = Tuple[float, float, float]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]


class ControleurCourbes(object):
    """ Gere un ensemble de courbes. """
    def __init__(self, vue_ref: Any = None) -> None:
        self.courbes: List[Modele.Courbe] = []
        self.scene: Import_scene.Donnees_scene | None = None
        self.zbuffer: Modele.ZBuffer | None = None
        self.camera = camera.Camera(distance=400)
        self.loaded_objects: List[Tuple[Import_scene.Polyedre, bool]] = []
        self.current_rendering_mode: str = 'fildefer' # Default rendering mode
        self.mode: str = 'viewer' # Initial mode: 'viewer' or 'edit'
        self.transformed_vertices_3d: List[vecteur3.Vecteur] = []
        self.projected_vertices_2d: List[Point2D] = []
        self.selected_vertex_index: Optional[Tuple[int, int]] = None # (object_index, vertex_index)
        self.vue_ref: Any = vue_ref # Reference to the Vue object for triggering updates. This is not ideal for strict MVC, but a practical shortcut for now.

    def rotate_camera(self, dx: float, dy: float) -> None:
        """Rotates the camera based on mouse movement."""
        self.camera.rotate(dx, dy)

    def zoom_camera(self, amount: float) -> None:
        """Zooms the camera in or out."""
        self.camera.zoom(amount)

    def ajouterCourbe(self, courbe: Modele.Courbe) -> None:
        """ Ajoute une courbe supplementaire.  """
        self.courbes.append(courbe)

    def dessiner(self, dessinerControle: DrawControlCallable, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine les courbes. """
        # Only RenderedTriangle implements remplir with zbuffer and scene
        # The other courbes (lines) will simply draw their points.
        for courbe in self.courbes:
            if isinstance(courbe, Modele.RenderedTriangle):
                if self.zbuffer is not None and self.scene is not None:
                    courbe.remplir(dessinerPoint, self.zbuffer, self.scene)
            else:
                courbe.dessinerPoints(dessinerPoint) # For wireframe lines and other basic courbes

        # dessine les points de controle
        # For now, only draw controls for non-triangle-filled primitives
        for courbe in self.courbes:
            if not isinstance(courbe, Modele.RenderedTriangle):
                courbe.dessinerControles(dessinerControle)

    def deplacerControle(self, ic: int, ip: int, point: Point2D) -> None:
        """ Deplace le point de controle a l'indice ip de la courbe a l'indice ic. """
        self.courbes[ic].controles[ip] = point

    def selectionnerControle(self, point: Point2D, mode: str) -> Callable[[Point2D], None] | None:
        """ Trouve un point de controle proche d'un point donne ou un sommet en mode edition. """
        xp, yp = point
        self.selected_vertex_index = None # Reset selected vertex

        if mode == 'edit' and self.loaded_objects:
            # Iterate through projected vertices to find a close one
            # Assuming a single loaded object for simplicity for now
            # TODO: Handle multiple objects and identify which object the vertex belongs to
            obj_idx = 0 
            if obj_idx < len(self.loaded_objects):
                obj, _ = self.loaded_objects[obj_idx]
                for v_idx, proj_v in enumerate(self.projected_vertices_2d):
                    # We need to map v_idx back to the object's listesommets index
                    # This assumes projected_vertices_2d is a direct concatenation of all objects' vertices
                    # which is currently true if only one object is loaded.
                    # For multiple objects, this logic needs refinement to correctly identify
                    # (object_index, vertex_index_within_object).
                    # For now, if there is one object, v_idx from projected_vertices_2d is also the index for listesommets
                    xc, yc = proj_v
                    if abs(xc - xp) < 8 and abs(yc - yp) < 8: # Increased hit radius for easier selection
                        self.selected_vertex_index = (obj_idx, v_idx)
                        print(f"Selected vertex: Object {obj_idx}, Vertex {v_idx}")
                        return lambda p: self._deplacerSommet(obj_idx, v_idx, p) # Return callable for moving vertex

        # Fallback to existing control point selection for 'viewer' mode or if no vertex selected in 'edit'
        for ic in range(len(self.courbes)):
            for ip in range(len(self.courbes[ic].controles)):
                xc, yc = self.courbes[ic].controles[ip]
                if abs(xc - xp) < 4 and abs(yc - yp) < 4:
                    return lambda p: self.deplacerControle(ic, ip, p)
        return None

    def _deplacerSommet(self, obj_idx: int, vertex_idx: int, new_2d_pos: Point2D) -> None:
        """ Deplace un sommet selectionne en mode edition. """
        if self.vue_ref is None: # Ensure vue_ref is available first
            print("Error: Vue reference not set in controller.")
            return
        
        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur

        if self.scene is None or not self.loaded_objects:
            return

        # Get the original Polyedre object and the vertex to move
        polyedre = self.loaded_objects[obj_idx][0]
        original_3d_vertex_coords = polyedre.listesommets[vertex_idx]
        original_3d_vertex = vecteur3.Vecteur(original_3d_vertex_coords[0], original_3d_vertex_coords[1], original_3d_vertex_coords[2])

        # Get current camera's view matrix and projection distance
        d = self.scene.d
        view_matrix = self.camera.get_view_matrix()
        inverse_view_matrix = view_matrix.inverse()

        # Get the 2D position of the vertex on screen before the move
        # We need this to calculate the delta in 2D for inverse projection
        if vertex_idx < len(self.projected_vertices_2d): # Check if projected vertex exists
            old_2d_pos = self.projected_vertices_2d[vertex_idx]
        else:
            print(f"Error: Projected vertex {vertex_idx} not found for object {obj_idx}.")
            return

        # Convert screen coordinates to relative coordinates for inverse projection
        # This undoes the (larg // 2 + x, (haut + 1) // 2 - 1 - y) transformation
        screen_x_old = old_2d_pos[0] - larg // 2
        screen_y_old = (haut + 1) // 2 - 1 - old_2d_pos[1]
        
        screen_x_new = new_2d_pos[0] - larg // 2
        screen_y_new = (haut + 1) // 2 - 1 - new_2d_pos[1]

        # To move a vertex in 3D based on 2D mouse movement, we need to decide its Z-depth.
        # A common approach is to keep its Z-depth relative to the camera constant during the drag.
        # So, we use the Z-depth of the original transformed vertex (in camera space).
        transformed_3d_vertex = self.transformed_vertices_3d[vertex_idx]
        current_camera_z = transformed_3d_vertex.vz
        
        target_camera_point = vecteur3.Vecteur(screen_x_new * current_camera_z / d,
                                               screen_y_new * current_camera_z / d,
                                               current_camera_z)

        # Transform this camera space point back to world space
        new_world_position = inverse_view_matrix.transform_point(target_camera_point)
        
        # Get the original object's center and its inverse translation matrix
        obj_center = polyedre.get_center()
        obj_inverse_center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(obj_center.vx, obj_center.vy, obj_center.vz))

        # Apply the full inverse transform to get the new local vertex position
        # v_local = inverse(center_translation) * inverse(view_matrix) * v_camera
        full_inverse_transform = obj_inverse_center_translation * inverse_view_matrix
        new_local_position = full_inverse_transform.transform_point(target_camera_point) # Use target_camera_point not new_camera_position directly

        # Update the vertex in the original listesommets
        polyedre.listesommets[vertex_idx][0] = new_local_position.vx
        polyedre.listesommets[vertex_idx][1] = new_local_position.vy
        polyedre.listesommets[vertex_idx][2] = new_local_position.vz

        # Re-render the scene
        self.rebuild_courbes(larg, haut)
        self.vue_ref.majAffichage()
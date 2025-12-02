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

        # Calculate the 2D delta in these relative coordinates
        delta_x_2d = screen_x_new - screen_x_old
        delta_y_2d = screen_y_new - screen_y_old

        # To move a vertex in 3D based on 2D mouse movement, we need to decide its Z-depth.
        # A common approach is to keep its Z-depth relative to the camera constant during the drag.
        # So, we use the Z-depth of the original transformed vertex (in camera space).
        transformed_3d_vertex = self.transformed_vertices_3d[vertex_idx]
        current_camera_z = transformed_3d_vertex.vz

        # Perform inverse perspective projection to get a 3D point in camera space
        # new_x_camera = (screen_x_new * current_camera_z) / d
        # new_y_camera = (screen_y_new * current_camera_z) / d
        # new_z_camera = current_camera_z # Keep Z constant

        # Let's calculate the delta in camera space based on 2D screen delta,
        # scaled by the current Z-depth and projection distance 'd'.
        # This assumes we are moving the point on a plane parallel to the screen at its current Z-depth.
        delta_x_camera = (delta_x_2d * current_camera_z) / d
        delta_y_camera = (delta_y_2d * current_camera_z) / d
        
        # Create a translation vector in camera space
        camera_space_translation = vecteur3.Vecteur(delta_x_camera, delta_y_camera, 0) # Only move in X, Y of camera plane

        # Transform this delta vector from camera space to world space
        # This is where we need the inverse of the view_matrix
        # It's better to transform the entire point rather than just a delta,
        # but for small movements, translating the current point by a world-space delta is also an option.
        # For direct manipulation, it's often more intuitive to apply the 2D screen delta
        # to the 3D point's projection plane.

        # Simplistic approach: directly apply a world-space delta based on camera delta
        # This isn't strictly correct inverse projection for arbitrary rotations, but a common approximation.
        # A more rigorous way would be to get the 3D world coords of the old_2d_pos and new_2d_pos
        # and subtract them.
        
        # We need the mouse move delta in screen space, then unproject this delta.
        # The change in screen space (dx_screen, dy_screen) corresponds to a change
        # in camera space (dx_camera, dy_camera) at the current Z depth.
        # dx_camera = dx_screen * current_camera_z / d
        # dy_camera = dy_screen * current_camera_z / d

        # To get the world space delta, we multiply by the inverse of the view matrix (rotation part).
        # We are simplifying and applying this delta directly to the vertex in its local space.

        # More robust inverse projection:
        # We have the 2D screen position (screen_x_new, screen_y_new) and current_camera_z.
        # We need to find the corresponding 3D world point.
        # Point in camera space (x_c, y_c, z_c) is given by:
        # x_c = screen_x_new * z_c / d
        # y_c = screen_y_new * z_c / d
        # z_c = current_camera_z (kept constant)
        
        target_camera_point = vecteur3.Vecteur(screen_x_new * current_camera_z / d,
                                               screen_y_new * current_camera_z / d,
                                               current_camera_z)

        # Transform this camera space point back to world space
        new_world_point_vec = inverse_view_matrix.transform_point(target_camera_point)
        
        # Get the original object's center and its inverse translation matrix
        obj_center = polyedre.get_center()
        obj_inverse_center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(obj_center.vx, obj_center.vy, obj_center.vz))

        # This inverse transformation should be applied carefully.
        # The new_world_point_vec is in world coordinates. We need to find
        # what its local coordinates would be relative to the object's origin.
        # This is essentially: local_new_point = inverse_obj_transform * new_world_point.
        # Given that we transformed points `v_transformed = center_translation * view_matrix * v`,
        # where `v` is the local vertex.
        # So, `v = inverse(center_translation * view_matrix) * v_transformed`.
        # `v = inverse_view_matrix * inverse_center_translation * v_transformed`.
        # This is complex because v_transformed is in camera space.

        # Let's reconsider. We want to move `original_3d_vertex` in its *local space*.
        # The camera space point `transformed_3d_vertex` corresponds to `original_3d_vertex`.
        # We have the desired `target_camera_point` in camera space.
        # The delta in camera space is `target_camera_point - transformed_3d_vertex`.
        # Let's call this `delta_camera_space`.

        delta_camera_space = target_camera_point - transformed_3d_vertex
        
        # Transform this delta from camera space to world space.
        # Only the rotational part of inverse_view_matrix affects delta (not translation)
        rotation_part_inverse_view = view_matrix.transposed() # For orthogonal matrices, inverse is transpose
        # More correctly, to transform a vector (delta) from camera to world:
        # We apply the inverse of the camera's orientation (rotation) to the delta.
        # If view_matrix = [R | t], then inverse_view_matrix = [R_T | -R_T * t].
        # For a vector, we only need R_T.
        
        # The `inverse_view_matrix` from matrix.py directly gives the inverse including translation.
        # To get the rotational inverse for a vector, we should use its rotational components.
        # A simple approximation is to use the inverse_view_matrix directly for the delta, assuming
        # the delta is small and we are moving it in the direction that corresponds to the camera's orientation.
        
        # Let's simplify for direct manipulation:
        # Calculate the 2D screen coordinates of the original_3d_vertex first
        # 1. Transform original_3d_vertex to camera space
        center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(-obj_center.vx, -obj_center.vy, -obj_center.vz))
        transform_matrix = center_translation * view_matrix
        v_transformed = transform_matrix.transform_point(original_3d_vertex)

        # 2. Project to 2D screen space
        if v_transformed.vz == 0:
            original_screen_x_relative = 0
            original_screen_y_relative = 0
        else:
            original_screen_x_relative = round(v_transformed.vx * d / v_transformed.vz)
            original_screen_y_relative = round(v_transformed.vy * d / v_transformed.vz)
        
        # Calculate the screen delta from original projected position to new_2d_pos
        delta_screen_x_relative = screen_x_new - original_screen_x_relative
        delta_screen_y_relative = screen_y_new - original_screen_y_relative

        # Now, we need to convert this 2D screen delta into a 3D world space delta.
        # This delta should be applied in world space, relative to the object's current position.
        # To maintain the Z depth of the vertex relative to the camera, we can calculate
        # how much a unit screen movement corresponds to a world movement at that Z.
        
        # A more straightforward approach:
        # We have the 2D screen delta. We can construct a 3D ray from the camera
        # through the original 2D point and the new 2D point, and find where they intersect
        # a plane (e.g., perpendicular to the camera's Z axis at the original vertex's Z).
        
        # For a simple 'grab' as in Blender, the movement is typically constrained
        # to a plane defined by the camera's view, passing through the selected vertex.
        # We can calculate a 3D world-space vector that corresponds to the 2D mouse movement.
        
        # Let's re-use the concept of target_camera_point but apply it as a movement in local object space.
        # The key is to correctly unproject the 2D mouse coordinates back into 3D world space.
        
        # Given a point (x_s, y_s) on screen and its current camera Z (z_c):
        # x_c = x_s * z_c / d
        # y_c = y_s * z_c / d
        
        # The movement is essentially moving the point on a plane parallel to the camera view plane.
        # The delta in camera coordinates:
        # delta_x_cam = (new_2d_pos[0] - old_2d_pos[0]) * current_camera_z / d
        # delta_y_cam = (new_2d_pos[1] - old_2d_pos[1]) * current_camera_z / d
        
        # This delta_cam is a vector in camera space.
        # We need to transform this vector into world space, which means applying the inverse rotation of the camera.
        
        # Let's represent the mouse movement as a vector in camera space:
        # We assume the mouse moves on a plane parallel to the camera's XY plane,
        # passing through the current 3D point (transformed_3d_vertex).
        
        # The transformation from world to camera space is `v_camera = view_matrix * v_world`.
        # So, `v_world = inverse(view_matrix) * v_camera`.

        # Let's compute the new camera space coordinates for the vertex, keeping its Z depth constant.
        new_camera_x = (screen_x_new * current_camera_z) / d
        new_camera_y = (screen_y_new * current_camera_z) / d
        
        new_camera_position = vecteur3.Vecteur(new_camera_x, new_camera_y, current_camera_z)
        
        # Transform this new camera position back to world coordinates
        new_world_position = inverse_view_matrix.transform_point(new_camera_position)

        # Now, new_world_position is the desired new world coordinate for the vertex.
        # However, the Polyedre.listesommets stores local coordinates relative to the object's center.
        # We need to apply the inverse of the object's 'center_translation' to new_world_position
        # to get its new local coordinates.

        # The transformation applied was: `v_transformed = (center_translation * view_matrix) * v_local`.
        # So, `v_local_new = inverse(center_translation * view_matrix) * new_camera_position`.
        # Which is `v_local_new = inverse_center_translation * inverse_view_matrix * new_camera_position`.
        
        # Calculate inverse center translation
        obj_inverse_center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(obj_center.vx, obj_center.vy, obj_center.vz))

        # Apply the full inverse transform to get the new local vertex position
        full_inverse_transform = obj_inverse_center_translation * inverse_view_matrix
        new_local_position = full_inverse_transform.transform_point(new_camera_position)

        # Update the vertex in the original listesommets
        polyedre.listesommets[vertex_idx][0] = new_local_position.vx
        polyedre.listesommets[vertex_idx][1] = new_local_position.vy
        polyedre.listesommets[vertex_idx][2] = new_local_position.vz

        # Re-render the scene
        # We need a reference to the Vue to call majAffichage.
        # The Controleur typically doesn't directly hold a reference to the Vue to avoid circular dependencies.
        # However, for simplicity in this exercise, we can add a weak reference or pass it.
        # For now, let's assume there's a way to trigger majAffichage, or majAffichage will be called externally.
        # In a typical MVC, the Controller would notify the View to update.
        # For testing, let's call rebuild_courbes directly.
        if hasattr(self, 'vue_ref') and self.vue_ref:
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
            self.vue_ref.majAffichage()




    def nouvelleHorizontale(self) -> Callable[[Point2D], None]:
        """ Ajoute une nouvelle horizontale initialement vide.
        Retourne une fonction permettant d'ajouter les points de controle. """
        horizontale = Modele.Horizontale()
        self.ajouterCourbe(horizontale)
        return horizontale.ajouterControle

    def nouvelleVerticale(self) -> Callable[[Point2D], None]:
        """ Ajoute une nouvelle verticale initialement vide.
        Retourne une fonction permettant d'ajouter les points de controle. """
        verticale = Modele.Verticale()
        self.ajouterCourbe(verticale)
        return verticale.ajouterControle

    def nouvelleGD(self) -> Callable[[Point2D], None]:
        """ Ajoute un segment gauche et un droite en meme temps initialement vide.
        Retourne une fonction permettant d'ajouter les points de controle. """
        seg_gauche_droite = Modele.GaucheDroite()
        self.ajouterCourbe(seg_gauche_droite)
        return seg_gauche_droite.ajouterControle

    def nouvellePointMilieu(self) -> Callable[[Point2D], None]:
        """ Initialise l'outils courant pour ajouter une nouvelle verticale. """
        seg_milieu = Modele.DroiteMilieu()
        self.ajouterCourbe(seg_milieu)
        return seg_milieu.ajouterControle



    def set_rendering_mode(self, larg: int, haut: int, mode: str) -> None:
        """Generates the courbes list for rendering based on loaded objects and selected mode."""
        self.current_rendering_mode = mode # Update the current rendering mode
        self.courbes = []  # Clear existing courbes for new rendering mode
        if self.scene is None: # self.scene must be set for object to be loaded
            return

        # Ensure zbuffer is initialized for zbuffer mode, or reset for other modes
        if mode == 'zbuffer':
            if self.zbuffer is None:
                self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else:
            self.zbuffer = None # Clear zbuffer for non-zbuffer modes

        d = self.scene.d
        view_matrix = self.camera.get_view_matrix()

        self.transformed_vertices_3d = []  # Clear global list
        self.projected_vertices_2d = []  # Clear global list
        self.courbes = []  # Clear existing courbes for new rendering mode

        if self.scene is None: # self.scene must be set for object to be loaded
            return

        # Ensure zbuffer is initialized for zbuffer mode, or reset for other modes
        if mode == 'zbuffer':
            if self.zbuffer is None:
                self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else:
            self.zbuffer = None # Clear zbuffer for non-zbuffer modes

        d = self.scene.d
        view_matrix = self.camera.get_view_matrix()

        # Populate global transformed and projected vertex lists once for all objects
        for indcptobj_stored, (obj, obj_texture) in enumerate(self.loaded_objects):
            # Recalculate transformations for each object
            center = obj.get_center()
            center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(-center.vx, -center.vy, -center.vz))
            transform_matrix = center_translation * view_matrix

            for som_coords in obj.listesommets:
                v = vecteur3.Vecteur(som_coords[0], som_coords[1], som_coords[2])
                v_transformed = transform_matrix.transform_point(v)
                self.transformed_vertices_3d.append(v_transformed) # Store as Vecteur

                # Project to 2D
                if v_transformed.vz == 0:
                    # Handle division by zero, possibly by clipping or placing far away
                    xp_2d = 0
                    yp_2d = 0
                else:
                    xp_2d = round(v_transformed.vx * d / v_transformed.vz)
                    yp_2d = round(v_transformed.vy * d / v_transformed.vz)
                self.projected_vertices_2d.append((larg // 2 + xp_2d, (haut + 1) // 2 - 1 - yp_2d)) # Store as Point2D

            # Iterate through faces and create RenderedTriangle instances
            # All modes will now use RenderedTriangle for consistent data access
            for i, _ in enumerate(obj.listeindicestriangle):
                if mode == 'fildefer':
                    # For wireframe, we still need lines. RenderedTriangle can draw a filled triangle,
                    # so we will use its vertex data to draw lines *around* it.
                    # This is a simplification; a proper wireframe might iterate edges.
                    # For now, we use the 3 vertices of the face to draw 3 lines.
                    v_indices = obj.listeindicestriangle[i]
                    p0_idx, p1_idx, p2_idx = v_indices[0] - 1, v_indices[1] - 1, v_indices[2] - 1

                    point0 = self.projected_vertices_2d[p0_idx]
                    point1 = self.projected_vertices_2d[p1_idx]
                    point2 = self.projected_vertices_2d[p2_idx]

                    droitemilieu1 = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu1)
                    droitemilieu1.ajouterControle(point0)
                    droitemilieu1.ajouterControle(point1)

                    droitemilieu2 = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu2)
                    droitemilieu2.ajouterControle(point1)
                    droitemilieu2.ajouterControle(point2)

                    droitemilieu3 = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu3)
                    droitemilieu3.ajouterControle(point2)
                    droitemilieu3.ajouterControle(point0)

                elif mode == 'peintre':
                    # For painter's algorithm, RenderedTriangle will act as the filled primitive
                    # We might need to sort these triangles later based on depth for actual painter's.
                    if self.zbuffer is None: # peintre mode does not use zbuffer normally
                        rendered_triangle = Modele.RenderedTriangle(obj, i, self.transformed_vertices_3d, self.projected_vertices_2d, self.scene, self.zbuffer)
                        self.ajouterCourbe(rendered_triangle)
                elif mode == 'zbuffer':
                    # Z-buffer mode directly uses RenderedTriangle
                    if self.zbuffer is not None:
                        rendered_triangle = Modele.RenderedTriangle(obj, i, self.transformed_vertices_3d, self.projected_vertices_2d, self.scene, self.zbuffer)
                        self.ajouterCourbe(rendered_triangle)


    def rebuild_courbes(self, larg: int, haut: int) -> None:
        """Rebuilds the list of courbes using the current rendering mode."""
        self.set_rendering_mode(larg, haut, self.current_rendering_mode)

    def importer_objet(self, larg: int, haut: int) -> None:
        """Imports an object and displays it in a default rendering mode."""
        # Initialize self.scene with scene data
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees

        fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
        if len(fic) > 0:
            indcptobj = 0 # Assuming only one object is loaded at a time into the scene's listeobjets
            obj_texture = donnees.ajoute_objet(fic, indcptobj)

            # Store a reference to this loaded object and its texture status
            self.loaded_objects = [(self.scene.listeobjets[indcptobj], obj_texture)] # Overwrite for single object mode

            # Render in a default mode (wireframe)
            self.current_rendering_mode = 'fildefer' # Set default mode
            self.rebuild_courbes(larg, haut)

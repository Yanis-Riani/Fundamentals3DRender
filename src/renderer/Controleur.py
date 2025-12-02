import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")

import tkinter.filedialog
from typing import Any, Callable, List, Tuple, Optional

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

        # Ensure zbuffer is initialized for zbuffer mode, or or reset for other modes
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

import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")

import tkinter.filedialog
import math
from typing import Any, Callable, List, Tuple, Optional, Dict

from ..camera import camera, matrix
from . import Import_scene, Modele, vecteur3

# Type aliases
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
        self.current_rendering_mode: str = 'fildefer'
        self.mode: str = 'viewer' # 'viewer' or 'edit'
        self.transformed_vertices_3d: List[vecteur3.Vecteur] = []
        self.projected_vertices_2d: List[Point2D] = []
        self.selected_vertex_index: Optional[Tuple[int, int]] = None # (object_index, vertex_index)
        self.vue_ref: Any = vue_ref 

        # State for Grab (Move) operation
        self.grab_state: Dict[str, Any] = {
            "active": False,
            "obj_idx": -1,
            "vertex_idx": -1,
            "original_pos": None, # Vecteur
            "constraint": None, # None, 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z'
        }

    def rotate_camera(self, dx: float, dy: float) -> None:
        """Rotates the camera."""
        self.camera.rotate(dx, dy)

    def zoom_camera(self, amount: float) -> None:
        """Zooms the camera."""
        self.camera.zoom(amount)

    def ajouterCourbe(self, courbe: Modele.Courbe) -> None:
        """ Ajoute une courbe supplementaire. """
        self.courbes.append(courbe)

    def dessiner(self, dessinerControle: DrawControlCallable, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine les courbes. """
        for courbe in self.courbes:
            if isinstance(courbe, Modele.RenderedTriangle):
                if self.zbuffer is not None and self.scene is not None:
                    courbe.remplir(dessinerPoint, self.zbuffer, self.scene)
            else:
                courbe.dessinerPoints(dessinerPoint)

        # Draw controls (only for non-filled primitives)
        for courbe in self.courbes:
            if not isinstance(courbe, Modele.RenderedTriangle):
                courbe.dessinerControles(dessinerControle)

    def selectionnerControle(self, point: Point2D, mode: str) -> None:
        """ 
        In 'edit' mode: Selects a vertex. 
        """
        xp, yp = point
        
        if mode == 'edit' and self.loaded_objects:
            self.selected_vertex_index = None # Reset selection
            
            # Simple hit detection
            obj_idx = 0 
            if obj_idx < len(self.loaded_objects):
                for v_idx, proj_v in enumerate(self.projected_vertices_2d):
                    xc, yc = proj_v
                    if abs(xc - xp) < 8 and abs(yc - yp) < 8:
                        self.selected_vertex_index = (obj_idx, v_idx)
                        print(f"Selected vertex: Object {obj_idx}, Vertex {v_idx}")
                        return

        # Legacy 2D curve selection removed as per instruction
        return None

    # --- Grab Mode Logic ---

    def start_grab_mode(self) -> None:
        """Initiates the grab mode for the selected vertex."""
        if self.mode != 'edit' or self.selected_vertex_index is None:
            return

        obj_idx, v_idx = self.selected_vertex_index
        if obj_idx >= len(self.loaded_objects): return
        
        obj = self.loaded_objects[obj_idx][0]
        v_coords = obj.listesommets[v_idx]
        
        self.grab_state["active"] = True
        self.grab_state["obj_idx"] = obj_idx
        self.grab_state["vertex_idx"] = v_idx
        self.grab_state["original_pos"] = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2])
        self.grab_state["constraint"] = None
        print("Grab mode started. Press 'X', 'Y', 'Z' (or Shift+) to constrain. Enter/Click to confirm. Esc/RightClick to cancel.")

    def confirm_grab(self) -> None:
        """Confirms the current grab operation."""
        if self.grab_state["active"]:
            self.grab_state["active"] = False
            print("Grab confirmed.")

    def cancel_grab(self) -> None:
        """Cancels the grab operation, reverting the vertex."""
        if self.grab_state["active"]:
            # Revert
            orig = self.grab_state["original_pos"]
            obj = self.loaded_objects[self.grab_state["obj_idx"]][0]
            obj.listesommets[self.grab_state["vertex_idx"]] = [orig.vx, orig.vy, orig.vz]
            
            self.grab_state["active"] = False
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
            self.vue_ref.majAffichage()
            print("Grab cancelled.")

    def toggle_axis_constraint(self, key: str, shift: bool) -> None:
        """Sets the axis constraint based on key press."""
        if not self.grab_state["active"]: return

        axis = key.lower()
        new_constraint = axis
        if shift:
            new_constraint = "shift_" + axis
        
        # Toggle off if same constraint selected again? (Optional, mimicking Blender)
        if self.grab_state["constraint"] == new_constraint:
            self.grab_state["constraint"] = None
            print("Constraint cleared.")
        else:
            self.grab_state["constraint"] = new_constraint
            print(f"Constraint set to: {new_constraint}")
        
        # We need to re-update the position using the last known mouse position.
        # However, Vue doesn't send it here. The next mouse move will update it.

    def update_grab(self, mouse_x: int, mouse_y: int) -> None:
        """Updates the vertex position based on mouse coordinates and constraints."""
        if not self.grab_state["active"]: return

        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur
        d = self.scene.d
        
        # 1. Calculate Ray from Camera
        # Screen rel coords
        # Correction of the Y axis to match the math in project/unproject
        # Screen Y is (haut + 1) // 2 - 1 - y_2d
        # So y_2d = (haut + 1) // 2 - 1 - Screen Y
        # And x_2d = Screen X - larg // 2
        
        scr_x_rel = mouse_x - larg // 2
        scr_y_rel = (haut + 1) // 2 - 1 - mouse_y
        
        # Camera Space Ray Direction (assuming perspective)
        # Point on projection plane at Z=d is (scr_x_rel, scr_y_rel, d) if we ignore scaling by Z?
        # In set_rendering_mode: xp_2d = v.vx * d / v.vz
        # So v.vx = xp_2d * v.vz / d
        # Ray direction D_cam = (scr_x_rel, scr_y_rel, d)
        ray_dir_cam = vecteur3.Vecteur(scr_x_rel, scr_y_rel, d).normer()
        ray_origin_cam = vecteur3.Vecteur(0, 0, 0)

        # 2. Transform Ray to World Space
        view_matrix = self.camera.get_view_matrix()
        inv_view = view_matrix.inverse()
        
        ray_origin_world = inv_view.transform_point(ray_origin_cam)
        # For direction, we transform a point on the ray and subtract origin
        ray_point_cam = ray_dir_cam 
        ray_point_world = inv_view.transform_point(ray_point_cam)
        
        ray_dir_world = (ray_point_world - ray_origin_world).normer()
        
        # 3. Handle Constraints
        orig_pos = self.grab_state["original_pos"] # World Space
        constraint = self.grab_state["constraint"]
        
        new_pos_world = orig_pos # Default
        
        # Helper: Intersect Ray with Plane(Normal, Point)
        def intersect_plane(normal: vecteur3.Vecteur, plane_point: vecteur3.Vecteur) -> Optional[vecteur3.Vecteur]:
            denom = normal.produitScalaire(ray_dir_world)
            if abs(denom) < 1e-6: return None
            t = (plane_point - ray_origin_world).produitScalaire(normal) / denom
            return ray_origin_world + (ray_dir_world * t)

        if constraint is None:
            # Free move: Move parallel to camera plane at original depth distance?
            # Or better: Plane defined by Camera Forward Vector passing through Original Point.
            # Camera Forward in World Space is inv_view * (0,0,1) - inv_view * (0,0,0) approx?
            # Actually, simpler: Use the plane perpendicular to the look direction.
            # Look direction is roughly ray_dir_world (or camera Z).
            # Let's use Camera Z axis in World Space as normal.
            cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
            new_pos_world = intersect_plane(cam_z_world, orig_pos)

        elif "shift_" in constraint:
            # Plane Constraint
            # shift_z -> Lock XY plane -> Normal is Z (0,0,1)
            axis_char = constraint.split('_')[1]
            normal = vecteur3.Vecteur(1 if axis_char=='x' else 0, 1 if axis_char=='y' else 0, 1 if axis_char=='z' else 0)
            intersection = intersect_plane(normal, orig_pos)
            if intersection:
                new_pos_world = intersection

        else:
            # Axis Constraint
            # Constraint is 'x', 'y', or 'z'
            axis_vec = vecteur3.Vecteur(1 if constraint=='x' else 0, 1 if constraint=='y' else 0, 1 if constraint=='z' else 0)
            
            # Technique: Intersect Ray with Plane defined by (Camera Z) passing through Orig,
            # Then project that point onto the Axis Line.
            cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
            
            # Plane intersection
            plane_hit = intersect_plane(cam_z_world, orig_pos)
            
            if plane_hit:
                # Project plane_hit onto line (Orig, AxisVec)
                # Proj P onto Line(A, D): A + dot(P-A, D) * D (if D normalized)
                diff = plane_hit - orig_pos
                dist = diff.produitScalaire(axis_vec)
                new_pos_world = orig_pos + (axis_vec * dist)
        
        if new_pos_world:
            # Update Model
            obj = self.loaded_objects[self.grab_state["obj_idx"]][0]
            
            # IMPORTANT: obj.listesommets is in LOCAL space. 
            # We calculated World Space.
            # Need to transform World -> Local.
            # Local = Inv(Translation(Center)) * World
            # Actually center is applied *before* view?
            # Code says: transform_matrix = center_translation * view_matrix
            # So World = Translation(-Center) * Local ? No.
            # Let's check set_rendering_mode:
            # center = obj.get_center()
            # center_translation = Translation(-center)
            # transform = center_translation * view_matrix
            # v_transformed = transform * v_local
            # So v_transformed (Camera Space) = View * Translation(-Center) * v_local
            # We have World Space (which is usually Inv(View) * CameraSpace).
            # So World Space = Translation(-Center) * v_local ? 
            # This implies the "World" is actually shifted by the object center relative to true world origin?
            # Or does Import_scene handle it?
            
            # Let's assume standard Model Matrix logic:
            # ModelMatrix = Translation(-Center)  (This centers the object at world origin 0,0,0)
            # So v_world = v_local - Center
            # v_local = v_world + Center
            
            center = obj.get_center()
            center_vec = vecteur3.Vecteur(center.vx, center.vy, center.vz)
            
            # But wait, original code in `_deplacerSommet`:
            # obj_inverse_center_translation = create_translation(obj_center)
            # full_inverse = obj_inverse_center_translation * inverse_view_matrix
            # new_local = full_inverse * target_camera_point
            
            # target_camera_point is Camera Space.
            # inverse_view_matrix * target_camera_point -> World Space (let's call it P_w)
            # obj_inverse_center_translation * P_w -> Local Space.
            # obj_inverse_center_translation is Translation(+Center).
            # So Local = World + Center.
            
            new_pos_local = new_pos_world + center_vec
            
            obj.listesommets[self.grab_state["vertex_idx"]] = [new_pos_local.vx, new_pos_local.vy, new_pos_local.vz]

            self.rebuild_courbes(larg, haut)
            self.vue_ref.majAffichage()


    def set_rendering_mode(self, larg: int, haut: int, mode: str) -> None:
        """Generates the courbes list for rendering based on loaded objects and selected mode."""
        self.current_rendering_mode = mode 
        self.courbes = []  
        if self.scene is None: 
            return

        if mode == 'zbuffer':
            if self.zbuffer is None:
                self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else:
            self.zbuffer = None

        d = self.scene.d
        view_matrix = self.camera.get_view_matrix()

        self.transformed_vertices_3d = []  
        self.projected_vertices_2d = [] 
        self.courbes = []  

        if self.scene is None:
            return

        if mode == 'zbuffer':
            if self.zbuffer is None:
                self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else:
            self.zbuffer = None 

        d = self.scene.d
        view_matrix = self.camera.get_view_matrix()

        for indcptobj_stored, (obj, obj_texture) in enumerate(self.loaded_objects):
            center = obj.get_center()
            center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(-center.vx, -center.vy, -center.vz))
            transform_matrix = center_translation * view_matrix

            for som_coords in obj.listesommets:
                v = vecteur3.Vecteur(som_coords[0], som_coords[1], som_coords[2])
                v_transformed = transform_matrix.transform_point(v)
                self.transformed_vertices_3d.append(v_transformed) 

                if v_transformed.vz == 0:
                    xp_2d = 0
                    yp_2d = 0
                else:
                    xp_2d = round(v_transformed.vx * d / v_transformed.vz)
                    yp_2d = round(v_transformed.vy * d / v_transformed.vz)
                self.projected_vertices_2d.append((larg // 2 + xp_2d, (haut + 1) // 2 - 1 - yp_2d)) 

            for i, _ in enumerate(obj.listeindicestriangle):
                if mode == 'fildefer':
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
                    rendered_triangle = Modele.RenderedTriangle(obj, i, self.transformed_vertices_3d, self.projected_vertices_2d, self.scene, None, larg, haut)
                    self.ajouterCourbe(rendered_triangle)
                elif mode == 'zbuffer':
                    if self.zbuffer is not None:
                        rendered_triangle = Modele.RenderedTriangle(obj, i, self.transformed_vertices_3d, self.projected_vertices_2d, self.scene, self.zbuffer, larg, haut)
                        self.ajouterCourbe(rendered_triangle)

    def rebuild_courbes(self, larg: int, haut: int) -> None:
        """Rebuilds the list of courbes using the current rendering mode."""
        self.set_rendering_mode(larg, haut, self.current_rendering_mode)

    def importer_objet(self, larg: int, haut: int) -> None:
        """Imports an object and displays it in a default rendering mode."""
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees

        fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
        if len(fic) > 0:
            indcptobj = 0
            obj_texture = donnees.ajoute_objet(fic, indcptobj)
            self.loaded_objects = [(self.scene.listeobjets[indcptobj], obj_texture)] 

            self.current_rendering_mode = 'fildefer' 
            self.rebuild_courbes(larg, haut)

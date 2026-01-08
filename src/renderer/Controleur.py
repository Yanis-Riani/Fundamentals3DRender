import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")

import tkinter.filedialog
import math
from typing import Any, Callable, List, Tuple, Optional, Dict, Set

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
        
        # Grid lines storage for optimized rendering: List[(x1, y1, x2, y2, color)]
        self.grid_lines_2d: List[Tuple[int, int, int, int, Couleur]] = []
        
        # Multi-selection: Set of (object_index, vertex_index)
        self.selected_vertices: Set[Tuple[int, int]] = set()
        
        self.vue_ref: Any = vue_ref 
        
        # Grid
        self.grid = Modele.Grille()

        # State for Transform (Grab/Rotate)
        self.transform_state: Dict[str, Any] = {
            "active": False,
            "mode": None, # 'grab' or 'rotate'
            "original_positions": {}, # Dict[(obj_idx, v_idx), Vecteur] (World Space)
            "pivot_original": None, # Vecteur (Centroid in World Space)
            "constraint": None, 
            "start_mouse_pos": (0, 0),
            "current_angle": 0.0
        }

        # Undo/Redo Stacks
        # Stores list of operations: { "type": "modify", "data": [((obj_idx, v_idx), old_pos, new_pos), ...] }
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

    def rotate_camera(self, dx: float, dy: float) -> None:
        self.camera.rotate(dx, dy)

    def zoom_camera(self, amount: float) -> None:
        self.camera.zoom(amount)
        
    def pan_camera(self, dx: float, dy: float) -> None:
        self.camera.pan(dx, dy)

    def ajouterCourbe(self, courbe: Modele.Courbe) -> None:
        self.courbes.append(courbe)

    def dessiner(self, dessinerControle: DrawControlCallable, dessinerPoint: DrawPointCallable) -> None:
        for courbe in self.courbes:
            if isinstance(courbe, Modele.RenderedTriangle):
                if self.zbuffer is not None and self.scene is not None:
                    courbe.remplir(dessinerPoint, self.zbuffer, self.scene)
            else:
                courbe.dessinerPoints(dessinerPoint)

        for courbe in self.courbes:
            if not isinstance(courbe, Modele.RenderedTriangle):
                courbe.dessinerControles(dessinerControle)

    def selectionnerControle(self, point: Point2D, mode: str, shift: bool = False) -> None:
        """ Selects a vertex. Handles Shift for toggle/add. """
        xp, yp = point
        
        if mode == 'edit' and self.loaded_objects:
            hit_found = None
            obj_idx = 0 
            if obj_idx < len(self.loaded_objects):
                for v_idx, proj_v in enumerate(self.projected_vertices_2d):
                    xc, yc = proj_v
                    if abs(xc - xp) < 8 and abs(yc - yp) < 8:
                        hit_found = (obj_idx, v_idx)
                        break
            
            if hit_found:
                if shift:
                    if hit_found in self.selected_vertices:
                        self.selected_vertices.remove(hit_found)
                    else:
                        self.selected_vertices.add(hit_found)
                else:
                    self.selected_vertices = {hit_found}
                print(f"Selection updated: {len(self.selected_vertices)} vertices.")
            else:
                if not shift:
                    self.selected_vertices.clear()

        return None

    def selectionner_zone(self, rect: Tuple[int, int, int, int], mode: str, shift: bool = False) -> None:
        """ Selects vertices within a 2D rectangle (Box Select). """
        if mode != 'edit' or not self.loaded_objects: return

        x1, y1, x2, y2 = rect
        # Normalize rect
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        obj_idx = 0
        new_selection = set()
        
        for v_idx, proj_v in enumerate(self.projected_vertices_2d):
            xc, yc = proj_v
            if x1 <= xc <= x2 and y1 <= yc <= y2:
                new_selection.add((obj_idx, v_idx))
        
        if shift:
            self.selected_vertices.update(new_selection)
        else:
            if new_selection: 
                self.selected_vertices = new_selection
            else:
                self.selected_vertices.clear()
        
        print(f"Box selection: {len(self.selected_vertices)} vertices.")


    # --- History Logic (Undo/Redo) ---

    def push_undo(self, operation: Dict[str, Any]) -> None:
        self.undo_stack.append(operation)
        self.redo_stack.clear() # New action clears redo
        print(f"Undo stack size: {len(self.undo_stack)}")

    def perform_undo(self) -> None:
        if not self.undo_stack:
            print("Nothing to undo.")
            return
        
        op = self.undo_stack.pop()
        self.redo_stack.append(op)
        
        if op["type"] == "transform":
            for (obj_idx, v_idx), old_pos, _ in op["data"]:
                self._update_vertex_position(obj_idx, v_idx, old_pos)
        
        self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
        self.vue_ref.majAffichage()
        print("Undo performed.")

    def perform_redo(self) -> None:
        if not self.redo_stack:
            print("Nothing to redo.")
            return
        
        op = self.redo_stack.pop()
        self.undo_stack.append(op)
        
        if op["type"] == "transform":
            for (obj_idx, v_idx), _, new_pos in op["data"]:
                self._update_vertex_position(obj_idx, v_idx, new_pos)

        self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
        self.vue_ref.majAffichage()
        print("Redo performed.")

    def get_undo_count(self) -> int:
        return len(self.undo_stack)

    def get_redo_count(self) -> int:
        return len(self.redo_stack)

    def _update_vertex_position(self, obj_idx: int, v_idx: int, pos_world: vecteur3.Vecteur) -> None:
        """Helper to update a vertex position given world coordinates."""
        obj = self.loaded_objects[obj_idx][0]
        obj.listesommets[v_idx] = [pos_world.vx, pos_world.vy, pos_world.vz]


    # --- Transform Mode Logic (Grab / Rotate) ---

    def start_transform_mode(self, mode: str, mouse_pos: Point2D) -> None:
        if self.mode != 'edit' or not self.selected_vertices:
            return

        self.transform_state["active"] = True
        self.transform_state["mode"] = mode
        self.transform_state["original_positions"] = {}
        self.transform_state["constraint"] = None
        self.transform_state["start_mouse_pos"] = mouse_pos
        self.transform_state["current_angle"] = 0.0
        
        pivot_accum = vecteur3.Vecteur(0,0,0)
        
        for obj_idx, v_idx in self.selected_vertices:
            obj = self.loaded_objects[obj_idx][0]
            v_coords = obj.listesommets[v_idx]
            v_world = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2])
            
            self.transform_state["original_positions"][(obj_idx, v_idx)] = v_world
            pivot_accum = pivot_accum + v_world
            
        # Calculate centroid as pivot
        count = len(self.selected_vertices)
        self.transform_state["pivot_original"] = pivot_accum * (1.0 / count)
        
        print(f"{mode.capitalize()} started on {count} vertices.")

    def confirm_transform(self) -> None:
        if self.transform_state["active"]:
            # Record History
            history_data = []
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items():
                obj = self.loaded_objects[obj_idx][0]
                v_coords = obj.listesommets[v_idx]
                curr_world = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2])
                
                history_data.append(((obj_idx, v_idx), orig_world, curr_world))
            
            self.push_undo({"type": "transform", "data": history_data})
            
            # Reset state
            self.transform_state["active"] = False
            self.transform_state["original_positions"] = {}
            print("Transform confirmed.")

    def cancel_transform(self) -> None:
        if self.transform_state["active"]:
            # Revert all
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items():
                self._update_vertex_position(obj_idx, v_idx, orig_world)
            
            self.transform_state["active"] = False
            self.transform_state["original_positions"] = {}
            
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
            self.vue_ref.majAffichage()
            print("Transform cancelled.")

    def toggle_axis_constraint(self, key: str, shift: bool) -> None:
        if not self.transform_state["active"]: return

        axis = key.lower()
        new_constraint = axis
        if shift:
            new_constraint = "shift_" + axis
        
        if self.transform_state["constraint"] == new_constraint:
            self.transform_state["constraint"] = None
            print("Constraint cleared.")
        else:
            self.transform_state["constraint"] = new_constraint
            print(f"Constraint set to: {new_constraint}")

    def update_transform(self, mouse_x: int, mouse_y: int) -> None:
        if not self.transform_state["active"]: return
        
        if self.transform_state["mode"] == 'grab':
            self._update_grab(mouse_x, mouse_y)
        elif self.transform_state["mode"] == 'rotate':
            self._update_rotate(mouse_x, mouse_y)

    def _get_mouse_ray(self, mouse_x: int, mouse_y: int) -> Tuple[vecteur3.Vecteur, vecteur3.Vecteur]:
        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur
        d = self.scene.d
        
        scr_x_rel = mouse_x - larg // 2
        scr_y_rel = (haut + 1) // 2 - 1 - mouse_y
        
        ray_dir_cam = vecteur3.Vecteur(scr_x_rel, scr_y_rel, d).normer()
        ray_origin_cam = vecteur3.Vecteur(0, 0, 0)

        view_matrix = self.camera.get_view_matrix()
        inv_view = view_matrix.inverse()
        
        ray_origin_world = inv_view.transform_point(ray_origin_cam)
        ray_point_world = inv_view.transform_point(ray_dir_cam)
        ray_dir_world = (ray_point_world - ray_origin_world).normer()
        
        return ray_origin_world, ray_dir_world

    def _intersect_plane(self, ray_origin: vecteur3.Vecteur, ray_dir: vecteur3.Vecteur, 
                         plane_normal: vecteur3.Vecteur, plane_point: vecteur3.Vecteur) -> Optional[vecteur3.Vecteur]:
        denom = plane_normal.produitScalaire(ray_dir)
        if abs(denom) < 1e-6: return None
        t = (plane_point - ray_origin).produitScalaire(plane_normal) / denom
        return ray_origin + (ray_dir * t)

    def _update_grab(self, mouse_x: int, mouse_y: int) -> None:
        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur
        
        # 1. Get Rays
        curr_ray_origin, curr_ray_dir = self._get_mouse_ray(mouse_x, mouse_y)
        start_x, start_y = self.transform_state["start_mouse_pos"]
        start_ray_origin, start_ray_dir = self._get_mouse_ray(start_x, start_y)
        
        pivot_orig = self.transform_state["pivot_original"]
        constraint = self.transform_state["constraint"]
        inv_view = self.camera.get_view_matrix().inverse()
        
        # Helper to get 3D point on constraint from a ray
        def get_projection_point(ray_origin: vecteur3.Vecteur, ray_dir: vecteur3.Vecteur) -> Optional[vecteur3.Vecteur]:
            if constraint is None:
                cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
                return self._intersect_plane(ray_origin, ray_dir, cam_z_world, pivot_orig)
            
            elif "shift_" in constraint:
                axis_char = constraint.split('_')[1]
                normal = vecteur3.Vecteur(1 if axis_char=='x' else 0, 1 if axis_char=='y' else 0, 1 if axis_char=='z' else 0)
                return self._intersect_plane(ray_origin, ray_dir, normal, pivot_orig)
            
            else:
                axis_vec = vecteur3.Vecteur(1 if constraint=='x' else 0, 1 if constraint=='y' else 0, 1 if constraint=='z' else 0)
                cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
                plane_hit = self._intersect_plane(ray_origin, ray_dir, cam_z_world, pivot_orig)
                
                if plane_hit:
                    diff = plane_hit - pivot_orig
                    dist = diff.produitScalaire(axis_vec)
                    return pivot_orig + (axis_vec * dist)
                return None

        hit_start = get_projection_point(start_ray_origin, start_ray_dir)
        hit_curr = get_projection_point(curr_ray_origin, curr_ray_dir)
        
        if hit_start and hit_curr:
            delta = hit_curr - hit_start
            
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items():
                new_pos_world = orig_world + delta
                self._update_vertex_position(obj_idx, v_idx, new_pos_world)

            self.rebuild_courbes(larg, haut)
            self.vue_ref.majAffichage()

    def _update_rotate(self, mouse_x: int, mouse_y: int) -> None:
        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur
        pivot_orig = self.transform_state["pivot_original"]
        constraint = self.transform_state["constraint"]
        view_matrix = self.camera.get_view_matrix()
        
        # 1. Project Pivot to Screen
        d = self.scene.d
        pivot_cam = view_matrix.transform_point(pivot_orig)
        
        if pivot_cam.vz == 0: return 
        
        pivot_screen_x = (larg // 2) + (pivot_cam.vx * d / pivot_cam.vz)
        pivot_screen_y = ((haut + 1) // 2) - 1 - (pivot_cam.vy * d / pivot_cam.vz)
        
        # 2. Calculate Angle
        start_x, start_y = self.transform_state["start_mouse_pos"]
        vec_start = (start_x - pivot_screen_x, start_y - pivot_screen_y)
        vec_curr = (mouse_x - pivot_screen_x, mouse_y - pivot_screen_y)
        
        angle_start = math.atan2(vec_start[1], vec_start[0])
        angle_curr = math.atan2(vec_curr[1], vec_curr[0])
        angle = -(angle_curr - angle_start)
        
        self.transform_state["current_angle"] = angle
        
        # 3. Determine Rotation Axis
        rot_axis = vecteur3.Vecteur(0,0,1) 
        
        inv_view = view_matrix.inverse()

        if constraint is None:
            cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
            rot_axis = cam_z_world
        else:
            axis_char = constraint[-1] 
            rot_axis = vecteur3.Vecteur(1 if axis_char=='x' else 0, 1 if axis_char=='y' else 0, 1 if axis_char=='z' else 0)

        # 4. Apply Rotation (Rodrigues)
        def rotate_point(point_rel: vecteur3.Vecteur, axis: vecteur3.Vecteur, theta: float) -> vecteur3.Vecteur:
            term1 = point_rel * math.cos(theta)
            term2 = axis.produitVectoriel(point_rel) * math.sin(theta)
            term3 = axis * (axis.produitScalaire(point_rel) * (1 - math.cos(theta)))
            return term1 + term2 + term3

        for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items():
            rel_vec = orig_world - pivot_orig
            rotated_rel = rotate_point(rel_vec, rot_axis, angle)
            new_pos_world = pivot_orig + rotated_rel
            self._update_vertex_position(obj_idx, v_idx, new_pos_world)

        self.rebuild_courbes(larg, haut)
        self.vue_ref.majAffichage()


    def set_rendering_mode(self, larg: int, haut: int, mode: str) -> None:
        self.current_rendering_mode = mode 
        self.courbes = []  
        self.grid_lines_2d = [] # Clear grid lines
        
        # Grid Rendering (Optimized)
        d = self.scene.d if self.scene else 200 
        view_matrix = self.camera.get_view_matrix()
        
        for p1_world, p2_world, color in self.grid.segments:
            p1_cam = view_matrix.transform_point(p1_world)
            p2_cam = view_matrix.transform_point(p2_world)
            
            # Clipping
            if p1_cam.vz <= 0 or p2_cam.vz <= 0: continue
            
            x1 = round(larg // 2 + p1_cam.vx * d / p1_cam.vz)
            y1 = round((haut + 1) // 2 - 1 - p1_cam.vy * d / p1_cam.vz)
            x2 = round(larg // 2 + p2_cam.vx * d / p2_cam.vz)
            y2 = round((haut + 1) // 2 - 1 - p2_cam.vy * d / p2_cam.vz)
            
            # Store directly instead of creating object
            self.grid_lines_2d.append((x1, y1, x2, y2, color))

        if self.scene is None: return

        if mode == 'zbuffer':
            if self.zbuffer is None:
                self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else:
            self.zbuffer = None

        d = self.scene.d
        self.transformed_vertices_3d = []  
        self.projected_vertices_2d = [] 

        for indcptobj_stored, (obj, obj_texture) in enumerate(self.loaded_objects):
            transform_matrix = view_matrix

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
        self.set_rendering_mode(larg, haut, self.current_rendering_mode)

    def importer_objet(self, larg: int, haut: int) -> None:
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees

        fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
        if len(fic) > 0:
            indcptobj = 0
            obj_texture = donnees.ajoute_objet(fic, indcptobj)
            
            obj = self.scene.listeobjets[indcptobj]
            center = obj.get_center()
            
            for i in range(len(obj.listesommets)):
                obj.listesommets[i][0] -= center.vx
                obj.listesommets[i][1] -= center.vy
                obj.listesommets[i][2] -= center.vz
            
            self.loaded_objects = [(obj, obj_texture)] 

            self.current_rendering_mode = 'fildefer' 
            self.rebuild_courbes(larg, haut)
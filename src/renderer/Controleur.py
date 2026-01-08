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
        
        # Multi-selection: Set of (object_index, vertex_index)
        self.selected_vertices: Set[Tuple[int, int]] = set()
        
        self.vue_ref: Any = vue_ref 

        # State for Grab (Move) operation
        self.grab_state: Dict[str, Any] = {
            "active": False,
            "original_positions": {}, # Dict[(obj_idx, v_idx), Vecteur] (World Space)
            "pivot_original": None, # Vecteur (Centroid in World Space)
            "constraint": None, 
        }

    def rotate_camera(self, dx: float, dy: float) -> None:
        self.camera.rotate(dx, dy)

    def zoom_camera(self, amount: float) -> None:
        self.camera.zoom(amount)

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
            if new_selection: # Only replace if we actually caught something? Or clear if empty? 
                # Standard behavior: Box select replaces unless shift is held.
                self.selected_vertices = new_selection
            else:
                self.selected_vertices.clear()
        
        print(f"Box selection: {len(self.selected_vertices)} vertices.")


    # --- Grab Mode Logic ---

    def start_grab_mode(self) -> None:
        if self.mode != 'edit' or not self.selected_vertices:
            return

        self.grab_state["active"] = True
        self.grab_state["original_positions"] = {}
        self.grab_state["constraint"] = None
        
        pivot_accum = vecteur3.Vecteur(0,0,0)
        
        for obj_idx, v_idx in self.selected_vertices:
            obj = self.loaded_objects[obj_idx][0]
            v_coords = obj.listesommets[v_idx]
            
            # Local to World
            center = obj.get_center()
            center_vec = vecteur3.Vecteur(center.vx, center.vy, center.vz)
            v_local = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2])
            v_world = v_local - center_vec
            
            self.grab_state["original_positions"][(obj_idx, v_idx)] = v_world
            pivot_accum = pivot_accum + v_world
            
        # Calculate centroid as pivot
        count = len(self.selected_vertices)
        self.grab_state["pivot_original"] = pivot_accum * (1.0 / count)
        
        print(f"Grab started on {count} vertices.")

    def confirm_grab(self) -> None:
        if self.grab_state["active"]:
            self.grab_state["active"] = False
            self.grab_state["original_positions"] = {}
            print("Grab confirmed.")

    def cancel_grab(self) -> None:
        if self.grab_state["active"]:
            # Revert all
            for (obj_idx, v_idx), orig_world in self.grab_state["original_positions"].items():
                obj = self.loaded_objects[obj_idx][0]
                center = obj.get_center()
                center_vec = vecteur3.Vecteur(center.vx, center.vy, center.vz)
                orig_local = orig_world + center_vec
                obj.listesommets[v_idx] = [orig_local.vx, orig_local.vy, orig_local.vz]
            
            self.grab_state["active"] = False
            self.grab_state["original_positions"] = {}
            
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur)
            self.vue_ref.majAffichage()
            print("Grab cancelled.")

    def toggle_axis_constraint(self, key: str, shift: bool) -> None:
        if not self.grab_state["active"]: return

        axis = key.lower()
        new_constraint = axis
        if shift:
            new_constraint = "shift_" + axis
        
        if self.grab_state["constraint"] == new_constraint:
            self.grab_state["constraint"] = None
            print("Constraint cleared.")
        else:
            self.grab_state["constraint"] = new_constraint
            print(f"Constraint set to: {new_constraint}")

    def update_grab(self, mouse_x: int, mouse_y: int) -> None:
        if not self.grab_state["active"]: return

        larg = self.vue_ref.largeur
        haut = self.vue_ref.hauteur
        d = self.scene.d
        
        # Ray casting
        scr_x_rel = mouse_x - larg // 2
        scr_y_rel = (haut + 1) // 2 - 1 - mouse_y
        
        ray_dir_cam = vecteur3.Vecteur(scr_x_rel, scr_y_rel, d).normer()
        ray_origin_cam = vecteur3.Vecteur(0, 0, 0)

        view_matrix = self.camera.get_view_matrix()
        inv_view = view_matrix.inverse()
        
        ray_origin_world = inv_view.transform_point(ray_origin_cam)
        ray_point_world = inv_view.transform_point(ray_dir_cam)
        ray_dir_world = (ray_point_world - ray_origin_world).normer()
        
        # Use Pivot for intersection logic
        pivot_orig = self.grab_state["pivot_original"]
        constraint = self.grab_state["constraint"]
        
        new_pivot_world = pivot_orig # Default
        
        def intersect_plane(normal: vecteur3.Vecteur, plane_point: vecteur3.Vecteur) -> Optional[vecteur3.Vecteur]:
            denom = normal.produitScalaire(ray_dir_world)
            if abs(denom) < 1e-6: return None
            t = (plane_point - ray_origin_world).produitScalaire(normal) / denom
            return ray_origin_world + (ray_dir_world * t)

        if constraint is None:
            cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
            new_pivot_world = intersect_plane(cam_z_world, pivot_orig)

        elif "shift_" in constraint:
            axis_char = constraint.split('_')[1]
            normal = vecteur3.Vecteur(1 if axis_char=='x' else 0, 1 if axis_char=='y' else 0, 1 if axis_char=='z' else 0)
            intersection = intersect_plane(normal, pivot_orig)
            if intersection:
                new_pivot_world = intersection

        else:
            axis_vec = vecteur3.Vecteur(1 if constraint=='x' else 0, 1 if constraint=='y' else 0, 1 if constraint=='z' else 0)
            cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
            plane_hit = intersect_plane(cam_z_world, pivot_orig)
            
            if plane_hit:
                diff = plane_hit - pivot_orig
                dist = diff.produitScalaire(axis_vec)
                new_pivot_world = pivot_orig + (axis_vec * dist)
        
        if new_pivot_world:
            # Calculate Delta
            delta = new_pivot_world - pivot_orig
            
            # Apply Delta to all selected vertices
            for (obj_idx, v_idx), orig_world in self.grab_state["original_positions"].items():
                obj = self.loaded_objects[obj_idx][0]
                
                # New World Pos
                new_pos_world = orig_world + delta
                
                # World to Local
                center = obj.get_center()
                center_vec = vecteur3.Vecteur(center.vx, center.vy, center.vz)
                new_pos_local = new_pos_world + center_vec
                
                obj.listesommets[v_idx] = [new_pos_local.vx, new_pos_local.vy, new_pos_local.vz]

            self.rebuild_courbes(larg, haut)
            self.vue_ref.majAffichage()


    def set_rendering_mode(self, larg: int, haut: int, mode: str) -> None:
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
        self.set_rendering_mode(larg, haut, self.current_rendering_mode)

    def importer_objet(self, larg: int, haut: int) -> None:
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
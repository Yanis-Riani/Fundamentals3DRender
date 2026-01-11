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
        
        # Optimized rendering lists
        self.grid_lines_2d: List[Tuple[int, int, int, int, Couleur]] = []
        self.wireframe_lines_2d: List[Tuple[int, int, int, int, Couleur]] = []
        self.solid_faces_2d: List[Tuple[List[Point2D], Couleur]] = []
        
        # Multi-selection: Set of (object_index, vertex_index)
        self.selected_vertices: Set[Tuple[int, int]] = set()
        self.visible_vertices: Set[int] = set()
        
        self.vue_ref: Any = vue_ref 
        
        # Grid Data
        self.grid_segments: List[Tuple[vecteur3.Vecteur, vecteur3.Vecteur, Couleur]] = []
        self._generate_grid_segments(2000.0, 100.0)

        # State for Transform
        self.transform_state: Dict[str, Any] = {
            "active": False,
            "mode": None, 
            "original_positions": {}, 
            "pivot_original": None, 
            "constraint": None, 
            "start_mouse_pos": (0, 0),
            "current_angle": 0.0
        }

        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

    def _generate_grid_segments(self, size: float, step: float) -> None:
        grey = (150, 150, 150)
        count = int(size / step)
        for i in range(-count, count + 1):
            coord = i * step
            self.grid_segments.append((vecteur3.Vecteur(coord, 0, -size), vecteur3.Vecteur(coord, 0, size), grey))
            self.grid_segments.append((vecteur3.Vecteur(-size, 0, coord), vecteur3.Vecteur(size, 0, coord), grey))
        self.grid_segments.append((vecteur3.Vecteur(-size, 0, 0), vecteur3.Vecteur(size, 0, 0), (255, 0, 0))) 
        self.grid_segments.append((vecteur3.Vecteur(0, 0, -size), vecteur3.Vecteur(0, 0, size), (0, 0, 255))) 

    def rotate_camera(self, dx: float, dy: float) -> None: self.camera.rotate(dx, dy)
    def zoom_camera(self, amount: float) -> None: self.camera.zoom(amount)
    def pan_camera(self, dx: float, dy: float) -> None: self.camera.pan(dx, dy)
    def ajouterCourbe(self, courbe: Modele.Courbe) -> None: self.courbes.append(courbe)

    def dessiner(self, dessinerControle: DrawControlCallable, dessinerPoint: DrawPointCallable) -> None:
        for courbe in self.courbes:
            if isinstance(courbe, Modele.RenderedTriangle):
                if self.zbuffer is not None and self.scene is not None: courbe.remplir(dessinerPoint, self.zbuffer, self.scene)
            else: courbe.dessinerPoints(dessinerPoint)
        for courbe in self.courbes:
            if not isinstance(courbe, Modele.RenderedTriangle): courbe.dessinerControles(dessinerControle)

    def selectionnerControle(self, point: Point2D, mode: str, shift: bool = False) -> None:
        xp, yp = point
        if mode == 'edit' and self.loaded_objects:
            hit_found = None; obj_idx = 0 
            if obj_idx < len(self.loaded_objects):
                for v_idx, proj_v in enumerate(self.projected_vertices_2d):
                    if self.current_rendering_mode in ['peintre', 'zbuffer'] and v_idx not in self.visible_vertices: continue
                    xc, yc = proj_v
                    if abs(xc - xp) < 8 and abs(yc - yp) < 8: hit_found = (obj_idx, v_idx); break
            if hit_found:
                if shift:
                    if hit_found in self.selected_vertices: self.selected_vertices.remove(hit_found)
                    else: self.selected_vertices.add(hit_found)
                else: self.selected_vertices = {hit_found}
            else:
                if not shift: self.selected_vertices.clear()
        return None

    def selectionner_zone(self, rect: Tuple[int, int, int, int], mode: str, shift: bool = False) -> None:
        if mode != 'edit' or not self.loaded_objects: return
        x1, y1, x2, y2 = rect
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        obj_idx = 0; new_selection = set()
        for v_idx, proj_v in enumerate(self.projected_vertices_2d):
            if self.current_rendering_mode in ['peintre', 'zbuffer'] and v_idx not in self.visible_vertices: continue
            xc, yc = proj_v
            if x1 <= xc <= x2 and y1 <= yc <= y2: new_selection.add((obj_idx, v_idx))
        if shift: self.selected_vertices.update(new_selection)
        else: self.selected_vertices = new_selection

    def push_undo(self, operation: Dict[str, Any]) -> None: self.undo_stack.append(operation); self.redo_stack.clear()

    def perform_undo(self) -> None:
        if not self.undo_stack: return
        op = self.undo_stack.pop(); self.redo_stack.append(op)
        if op["type"] == "transform":
            for (obj_idx, v_idx), old_pos, _ in op["data"]: self._update_vertex_position(obj_idx, v_idx, old_pos)
        self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur); self.vue_ref.majAffichage()

    def perform_redo(self) -> None:
        if not self.redo_stack: return
        op = self.redo_stack.pop(); self.undo_stack.append(op)
        if op["type"] == "transform":
            for (obj_idx, v_idx), _, new_pos in op["data"]: self._update_vertex_position(obj_idx, v_idx, new_pos)
        self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur); self.vue_ref.majAffichage()

    def get_undo_count(self) -> int: return len(self.undo_stack)
    def get_redo_count(self) -> int: return len(self.redo_stack)

    def _update_vertex_position(self, obj_idx: int, v_idx: int, pos_world: vecteur3.Vecteur) -> None:
        obj = self.loaded_objects[obj_idx][0]; obj.listesommets[v_idx] = [pos_world.vx, pos_world.vy, pos_world.vz]

    def start_transform_mode(self, mode: str, mouse_pos: Point2D) -> None:
        if self.mode != 'edit' or not self.selected_vertices: return
        self.transform_state.update({"active": True, "mode": mode, "original_positions": {}, "constraint": None, "start_mouse_pos": mouse_pos, "current_angle": 0.0})
        pivot_accum = vecteur3.Vecteur(0,0,0)
        for obj_idx, v_idx in self.selected_vertices:
            v_coords = self.loaded_objects[obj_idx][0].listesommets[v_idx]; v_world = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2])
            self.transform_state["original_positions"][(obj_idx, v_idx)] = v_world; pivot_accum = pivot_accum + v_world
        self.transform_state["pivot_original"] = pivot_accum * (1.0 / len(self.selected_vertices))

    def confirm_transform(self) -> None:
        if self.transform_state["active"]:
            history_data = []
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items():
                v_coords = self.loaded_objects[obj_idx][0].listesommets[v_idx]; curr_world = vecteur3.Vecteur(v_coords[0], v_coords[1], v_coords[2]); history_data.append(((obj_idx, v_idx), orig_world, curr_world))
            self.push_undo({"type": "transform", "data": history_data}); self.transform_state["active"] = False; self.transform_state["original_positions"] = {}

    def cancel_transform(self) -> None:
        if self.transform_state["active"]:
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items(): self._update_vertex_position(obj_idx, v_idx, orig_world)
            self.transform_state["active"] = False; self.transform_state["original_positions"] = {}
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur); self.vue_ref.majAffichage()

    def toggle_axis_constraint(self, key: str, shift: bool) -> None:
        if not self.transform_state["active"]: return
        new_c = key.lower() if not shift else "shift_" + key.lower()
        self.transform_state["constraint"] = None if self.transform_state["constraint"] == new_c else new_c

    def update_transform(self, mouse_x: int, mouse_y: int) -> None:
        if not self.transform_state["active"]: return
        if self.transform_state["mode"] == 'grab': self._update_grab(mouse_x, mouse_y)
        elif self.transform_state["mode"] == 'rotate': self._update_rotate(mouse_x, mouse_y)

    def _get_mouse_ray(self, mouse_x: int, mouse_y: int) -> Tuple[vecteur3.Vecteur, vecteur3.Vecteur]:
        scr_x_rel, scr_y_rel = mouse_x - self.vue_ref.largeur // 2, (self.vue_ref.hauteur + 1) // 2 - 1 - mouse_y
        d = self.scene.d if self.scene else 400
        ray_dir_cam = vecteur3.Vecteur(scr_x_rel, scr_y_rel, d).normer(); inv_view = self.camera.get_view_matrix().inverse()
        ray_origin_world = inv_view.transform_point(vecteur3.Vecteur(0, 0, 0)); ray_dir_world = (inv_view.transform_point(ray_dir_cam) - ray_origin_world).normer()
        return ray_origin_world, ray_dir_world

    def _intersect_plane(self, ray_origin: vecteur3.Vecteur, ray_dir: vecteur3.Vecteur, plane_normal: vecteur3.Vecteur, plane_point: vecteur3.Vecteur) -> Optional[vecteur3.Vecteur]:
        denom = plane_normal.produitScalaire(ray_dir)
        if abs(denom) < 1e-6: return None
        return ray_origin + (ray_dir * ((plane_point - ray_origin).produitScalaire(plane_normal) / denom))

    def _clip_line_near(self, p1: vecteur3.Vecteur, p2: vecteur3.Vecteur, near_z: float) -> Optional[Tuple[vecteur3.Vecteur, vecteur3.Vecteur]]:
        if p1.vz < near_z and p2.vz < near_z: return None
        if p1.vz >= near_z and p2.vz >= near_z: return (p1, p2)
        pi = p1 + (p2 - p1) * ((near_z - p1.vz) / (p2.vz - p1.vz))
        return (pi, p2) if p1.vz < near_z else (p1, pi)

    def _update_grab(self, mouse_x: int, mouse_y: int) -> None:
        curr_ray_o, curr_ray_d = self._get_mouse_ray(mouse_x, mouse_y); start_ray_o, start_ray_d = self._get_mouse_ray(*self.transform_state["start_mouse_pos"])
        pivot_orig, constraint = self.transform_state["pivot_original"], self.transform_state["constraint"]; inv_view = self.camera.get_view_matrix().inverse()
        def get_proj(ro, rd):
            if constraint is None: cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer(); return self._intersect_plane(ro, rd, cam_z_world, pivot_orig)
            elif "shift_" in constraint: ax = constraint.split('_')[1]; norm = vecteur3.Vecteur(1 if ax=='x' else 0, 1 if ax=='y' else 0, 1 if ax=='z' else 0); return self._intersect_plane(ro, rd, norm, pivot_orig)
            else:
                ax_v = vecteur3.Vecteur(1 if constraint=='x' else 0, 1 if constraint=='y' else 0, 1 if constraint=='z' else 0); cam_z_world = (inv_view.transform_point(vecteur3.Vecteur(0,0,1)) - inv_view.transform_point(vecteur3.Vecteur(0,0,0))).normer()
                hit = self._intersect_plane(ro, rd, cam_z_world, pivot_orig); return pivot_orig + (ax_v * (hit - pivot_orig).produitScalaire(ax_v)) if hit else None
        h_start, h_curr = get_proj(start_ray_o, start_ray_d), get_proj(curr_ray_o, curr_ray_d)
        if h_start and h_curr:
            delta = h_curr - h_start
            for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items(): self._update_vertex_position(obj_idx, v_idx, orig_world + delta)
            self.rebuild_courbes(self.vue_ref.largeur, self.vue_ref.hauteur); self.vue_ref.majAffichage()

    def _update_rotate(self, mouse_x: int, mouse_y: int) -> None:
        larg, haut = self.vue_ref.largeur, self.vue_ref.hauteur; pivot_orig, constraint = self.transform_state["pivot_original"], self.transform_state["constraint"]; view_matrix = self.camera.get_view_matrix(); d = self.scene.d if self.scene else 400
        p_cam = view_matrix.transform_point(pivot_orig)
        if p_cam.vz == 0: return
        p_scr_x, p_scr_y = (larg // 2) + (p_cam.vx * d / p_cam.vz), ((haut + 1) // 2) - 1 - (p_cam.vy * d / p_cam.vz); start_x, start_y = self.transform_state["start_mouse_pos"]
        angle = -(math.atan2(mouse_y - p_scr_y, mouse_x - p_scr_x) - math.atan2(start_y - p_scr_y, start_x - p_scr_x)); self.transform_state["current_angle"] = angle
        inv_v = view_matrix.inverse()
        if constraint is None: rot_axis = (inv_v.transform_point(vecteur3.Vecteur(0,0,1)) - inv_v.transform_point(vecteur3.Vecteur(0,0,0))).normer()
        else: ax = constraint[-1]; rot_axis = vecteur3.Vecteur(1 if ax=='x' else 0, 1 if ax=='y' else 0, 1 if ax=='z' else 0)
        def rot_p(p_rel, axis, theta): return p_rel * math.cos(theta) + axis.produitVectoriel(p_rel) * math.sin(theta) + axis * (axis.produitScalaire(p_rel) * (1 - math.cos(theta)))
        for (obj_idx, v_idx), orig_world in self.transform_state["original_positions"].items(): self._update_vertex_position(obj_idx, v_idx, pivot_orig + rot_p(orig_world - pivot_orig, rot_axis, angle))
        self.rebuild_courbes(larg, haut); self.vue_ref.majAffichage()

    def set_rendering_mode(self, larg: int, haut: int, mode: str) -> None:
        self.current_rendering_mode, self.courbes, self.grid_lines_2d, self.wireframe_lines_2d, self.solid_faces_2d = mode, [], [], [], []
        step, grid_range, fog_start, fog_end, bg_rgb, near_z = 100.0, 2000.0, 500.0, 2000.0, (211, 211, 211), 1.0
        target = self.camera.target; center_x, center_z = round(target.vx / step) * step, round(target.vz / step) * step
        start_x, end_x, start_z, end_z = center_x - grid_range, center_x + grid_range, center_z - grid_range, center_z + grid_range
        view_matrix, d = self.camera.get_view_matrix(), (self.scene.d if self.scene else 400)
        def proc_seg(w1, w2, bcol):
            c1, c2 = view_matrix.transform_point(w1), view_matrix.transform_point(w2); clipped = self._clip_line_near(c1, c2, near_z)
            if not clipped: return
            c1, c2 = clipped; dist = ((w1 + w2) * 0.5 - self.camera.position).norm()
            if dist > fog_end: return
            f = 1.0 if dist <= fog_start else 1.0 - (dist - fog_start) / (fog_end - fog_start)
            col = tuple(int(bcol[i] * f + bg_rgb[i] * (1 - f)) for i in range(3))
            self.grid_lines_2d.append((round(larg//2 + c1.vx*d/c1.vz), round((haut+1)//2 - 1 - c1.vy*d/c1.vz), round(larg//2 + c2.vx*d/c2.vz), round((haut+1)//2 - 1 - c2.vy*d/c2.vz), col))
        for z in range(int(start_z), int(end_z) + int(step), int(step)):
            bcol = (0, 0, 255) if z == 0 else (150, 150, 150)
            for x in range(int(start_x), int(end_x), int(step)): proc_seg(vecteur3.Vecteur(x, 0, z), vecteur3.Vecteur(x + step, 0, z), bcol)
        for x in range(int(start_x), int(end_x) + int(step), int(step)):
            bcol = (255, 0, 0) if x == 0 else (150, 150, 150)
            for z in range(int(start_z), int(end_z), int(step)): proc_seg(vecteur3.Vecteur(x, 0, z), vecteur3.Vecteur(x, 0, z + step), bcol)
        if self.scene is None: return
        if mode == 'zbuffer':
            if self.zbuffer is None: self.zbuffer = Modele.ZBuffer()
            self.zbuffer.alloc_init_zbuffer(larg, haut)
        else: self.zbuffer = None
        self.transformed_vertices_3d, self.projected_vertices_2d = [], []
        self.visible_vertices = set()
        vertex_offset = 0
        use_culling, faces_to_render = (mode in ['peintre', 'zbuffer']), []
        for obj, obj_tex in self.loaded_objects:
            obj_vertices_cam, obj_projected_2d = [], []
            for som in obj.listesommets:
                v_t = view_matrix.transform_point(vecteur3.Vecteur(*som)); obj_vertices_cam.append(v_t)
                xp, yp = (0, 0) if v_t.vz == 0 else (round(v_t.vx*d/v_t.vz), round(v_t.vy*d/v_t.vz))
                proj = (larg // 2 + xp, (haut + 1) // 2 - 1 - yp); obj_projected_2d.append(proj)
                self.transformed_vertices_3d.append(v_t); self.projected_vertices_2d.append(proj)
            for i in range(len(obj.listeindicestriangle)):
                v_i = obj.listeindicestriangle[i]; p0_cam, p1_cam, p2_cam = [obj_vertices_cam[idx-1] for idx in v_i]
                if not use_culling or (p1_cam - p0_cam).produitVectoriel(p2_cam - p0_cam).produitScalaire(-p0_cam) > 0:
                    faces_to_render.append({'obj': obj, 'face_idx': i, 'z': (p0_cam.vz + p1_cam.vz + p2_cam.vz) / 3.0, 'p2d': [obj_projected_2d[idx-1] for idx in v_i], 'color': obj.listecouleurs[i]})
                    if use_culling:
                        for idx in v_i: self.visible_vertices.add(vertex_offset + idx - 1)
            vertex_offset += len(obj.listesommets)
        faces_to_render.sort(key=lambda f: f['z'], reverse=True)
        for f in faces_to_render:
            if mode == 'fildefer':
                for start, end in [(f['p2d'][0], f['p2d'][1]), (f['p2d'][1], f['p2d'][2]), (f['p2d'][2], f['p2d'][0])]: self.wireframe_lines_2d.append((start[0], start[1], end[0], end[1], (0,0,0)))
            elif mode == 'peintre': 
                self.solid_faces_2d.append((f['p2d'], f['color']))
                # If in edit mode, also draw wireframe over solid faces
                if self.mode == 'edit':
                    for start, end in [(f['p2d'][0], f['p2d'][1]), (f['p2d'][1], f['p2d'][2]), (f['p2d'][2], f['p2d'][0])]: self.wireframe_lines_2d.append((start[0], start[1], end[0], end[1], (100,100,100)))
            elif mode == 'zbuffer':
                self.ajouterCourbe(Modele.RenderedTriangle(f['obj'], f['face_idx'], self.transformed_vertices_3d, self.projected_vertices_2d, self.scene, self.zbuffer, larg, haut))
                # Note: For Z-Buffer, we can't easily overlay wireframe here as it's drawn later viacurves. 
                # But we can add wireframe lines to curves or specialized list.
                if self.mode == 'edit':
                    for start, end in [(f['p2d'][0], f['p2d'][1]), (f['p2d'][1], f['p2d'][2]), (f['p2d'][2], f['p2d'][0])]: self.wireframe_lines_2d.append((start[0], start[1], end[0], end[1], (100,100,100)))

    def rebuild_courbes(self, larg: int, haut: int) -> None: self.set_rendering_mode(larg, haut, self.current_rendering_mode)

    def importer_objet(self, larg: int, haut: int) -> None:
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees
        fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"), filetypes=[("Fichiers Objets", "*.obj")])
        if fic:
            obj_idx = 0; obj_tex = donnees.ajoute_objet(fic, obj_idx); obj = self.scene.listeobjets[obj_idx]; center = obj.get_center()
            for i in range(len(obj.listesommets)): obj.listesommets[i][0] -= center.vx; obj.listesommets[i][1] -= center.vy; obj.listesommets[i][2] -= center.vz
            self.loaded_objects = [(obj, obj_tex)]; self.current_rendering_mode = 'fildefer'; self.rebuild_courbes(larg, haut)
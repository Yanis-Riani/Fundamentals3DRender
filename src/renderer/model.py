from __future__ import annotations
from typing import List, Tuple, Callable, Any, Optional

from . import edge
from . import vector3
# This import is just for type annotation to avoid circular dependencies at runtime
from .import_scene import SceneData, Polyhedron

# Type aliases for clarity
Point2D = Tuple[int, int]
Point3D = Tuple[float, float, float]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]


class Curve(object):
    """ Generic base class for a curve. """

    def __init__(self) -> None:
        self.controls: List[Point2D] = []

    def draw_controls(self, draw_control: DrawControlCallable) -> None:
        """ Draws the control points of the curve. """
        for control in self.controls:
            draw_control(control)

    def draw_points(self, draw_point: DrawPointCallable) -> None:
        """ Draws the curve. Method to be redefined in derived classes. """
        pass

    def add_control(self, point: Point2D) -> None:
        """ Adds a control point. """
        self.controls.append(point)

    def fill(self, draw_point: DrawPointCallable, *args: Any) -> None:
        """ Fill a closed curve (triangle). """
        pass


class Horizontal(Curve):
    """ Defines a horizontal line. """

    def add_control(self, point: Point2D) -> None:
        if len(self.controls) < 2:
            super().add_control(point)

    def draw_points(self, draw_point: DrawPointCallable) -> None:
        if len(self.controls) == 2:
            x1 = self.controls[0][0]
            x2 = self.controls[1][0]
            y = self.controls[0][1]
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            for x in range(x_min, x_max):
                draw_point((x, y), (0, 0, 0))


class Vertical(Curve):
    """ Defines a vertical line. """

    def add_control(self, point: Point2D) -> None:
        if len(self.controls) < 2:
            super().add_control(point)

    def draw_points(self, draw_point: DrawPointCallable) -> None:
        if len(self.controls) == 2:
            x = self.controls[0][0]
            y1 = self.controls[0][1]
            y2 = self.controls[1][1]
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            for y in range(y_min, y_max):
                draw_point((x, y), (0, 200, 200))


class LeftRight(Curve):
    """ Defines left/right edges (Legacy logic). """

    def add_control(self, point: Point2D) -> None:
        if len(self.controls) < 2:
            super().add_control(point)

    def draw_points(self, draw_point: DrawPointCallable) -> None:
        if len(self.controls) == 2:
            x1 = self.controls[0][0]
            x2 = self.controls[1][0]
            y1 = self.controls[0][1]
            y2 = self.controls[1][1]

            if (y1 > y2):
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            num = x2 - x1
            den = y2 - y1
            if den == 0: return
            inc = 0
            x = x1
            y = y1

            if (num > 0):
                inc = den - 1

            while (y < y2):
                draw_point((x, y), (0, 255, 0))
                inc += num
                q = inc // den
                x += q
                inc -= q * den
                y += 1

            # Right edge
            inc = -den
            x = x1
            y = y1

            if (num > 0):
                inc = -1

            while (y < y2):
                draw_point((x, y), (255, 0, 0))
                inc += num
                q = inc // den
                x += q
                inc -= q * den
                y += 1


class MidpointLine(Curve):
    """ Defines a line using the midpoint algorithm. """
    def __init__(self, color: Couleur = (0, 0, 0)) -> None:
        super().__init__()
        self.color = color

    def add_control(self, point: Point2D) -> None:
        if len(self.controls) < 2:
            super().add_control(point)

    def draw_points(self, draw_point: DrawPointCallable) -> None:
        if len(self.controls) == 2:
            x1, y1 = self.controls[0]
            x2, y2 = self.controls[1]

            if y1 > y2:  # Swap points
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            
            dy = y2 - y1  # dy > 0

            if (x2 >= x1):
                dx = x2 - x1
                y = y1
                if (dx >= dy):
                    dp = 2 * dy - dx
                    delta_e = 2 * dy
                    delta_ne = 2 * (dy - dx)
                    draw_point((x1, y1), self.color)
                    for x in range(x1, x2):
                        if (dp <= 0):
                            dp += delta_e
                        else:
                            dp += delta_ne
                            y += 1
                        draw_point((x, y), self.color)
                else:
                    dp = 2 * dx - dy
                    delta_e = 2 * dx
                    delta_ne = 2 * (dx - dy)
                    draw_point((x1, y1), self.color)
                    x = x1
                    for y in range(y1, y2):
                        if (dp <= 0):
                            dp += delta_e
                        else:
                            dp += delta_ne
                            x += 1
                        draw_point((x, y), self.color)
            else:
                dx = x1 - x2
                y = y1
                if (dx >= dy):
                    dp = 2 * dy - dx
                    delta_e = 2 * dy
                    delta_ne = 2 * (dy - dx)
                    draw_point((x1, y1), self.color)
                    for x in range(x1, x2, -1):
                        if (dp <= 0):
                            dp += delta_e
                        else:
                            dp += delta_ne
                            y += 1
                        draw_point((x, y), self.color)
                else:
                    dp = 2 * dx - dy
                    delta_e = 2 * dx
                    delta_ne = 2 * (dx - dy)
                    draw_point((x1, y1), self.color)
                    x = x1
                    for y in range(y1, y2):
                        if (dp <= 0):
                            dp += delta_e
                        else:
                            dp += delta_ne
                            x -= 1
                        draw_point((x, y), self.color)


class ZBuffer():
    def __init__(self) -> None:
        self.buffer: List[List[float]] = []
        self.dim_x: int = 0
        self.dim_y: int = 0

    def init_buffer(self, width: int, height: int) -> None:
        """Initializes the Z-buffer."""
        self.buffer = [[100000.0] * height for _ in range(width)]
        self.dim_x = width
        self.dim_y = height

    def get(self, i: int, j: int) -> float:
        return self.buffer[i][j]

    def set(self, i: int, j: int, val: float) -> None:
        self.buffer[i][j] = val


class FaceProperties():
    def __init__(self) -> None:
        self.normal_and_plane: List[float] = []
        self.texture_on: bool = False
        self.color: Couleur = (0, 0, 0)
        self.coeffs: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
        self.object_index: int = -1

class RenderedTriangle(Curve):
    def __init__(self, polyhedron_ref: Polyhedron, triangle_face_index: int,
                 transformed_vertices_3d: List[vector3.Vector3],
                 projected_vertices_2d: List[Point2D],
                 scene_ref: SceneData,
                 zbuffer_ref: Optional[ZBuffer],
                 width: int, height: int) -> None:
        super().__init__()
        self.polyhedron = polyhedron_ref
        self.triangle_face_index = triangle_face_index
        self.transformed_vertices_3d = transformed_vertices_3d
        self.projected_vertices_2d = projected_vertices_2d
        self.scene = scene_ref
        self.zbuffer = zbuffer_ref
        self.width = width 
        self.height = height

        self.vertex_indices = self.polyhedron.triangle_indices[self.triangle_face_index]
        self.normal_indices = self.polyhedron.normal_indices[self.triangle_face_index]
        self.texture_indices = self.polyhedron.texture_indices[self.triangle_face_index]
        self.face_color = self.polyhedron.colors[self.triangle_face_index]
        self.face_coeffs = self.polyhedron.coeffs[self.triangle_face_index]
        self.face_texture_on = self.polyhedron.texture_on
        
        self.face_properties = FaceProperties()
        self.face_properties.color = self.face_color
        self.face_properties.coeffs = self.face_coeffs
        self.face_properties.texture_on = self.face_texture_on
        self.face_properties.object_index = self.polyhedron.object_index

    def sort_by_y(self, y1: float, y2: float, y3: float) -> List[int]:
        if (y1 <= y2):
            if (y2 <= y3):
                return [0, 1, 2]
            else:
                if (y1 <= y3):
                    return [0, 2, 1]
                else:
                    return [2, 0, 1]
        else:
            if (y2 > y3):
                return [2, 1, 0]
            else:
                if (y3 > y1):
                    return [1, 0, 2]
                else:
                    return [1, 2, 0]

    def _get_vertex_data(self, local_vertex_index: int) -> Tuple[Point2D, vector3.Vector3, vector3.Vector3, Tuple[float, float]]:
        v_idx_0based = self.vertex_indices[local_vertex_index] - 1
        n_idx_0based = (self.normal_indices[local_vertex_index] - 1) if self.normal_indices else -1
        t_idx_0based = (self.texture_indices[local_vertex_index] - 1) if self.texture_indices else -1

        p2d = self.projected_vertices_2d[v_idx_0based]
        p3d = self.transformed_vertices_3d[v_idx_0based]

        normal_data = self.polyhedron.normals[n_idx_0based] if 0 <= n_idx_0based < len(self.polyhedron.normals) else [0.0, 0.0, 0.0]
        normal = vector3.Vector3(normal_data[0], normal_data[1], normal_data[2])

        tex_coord = self.polyhedron.texture_coords[t_idx_0based] if 0 <= t_idx_0based < len(self.polyhedron.texture_coords) else (0.0, 0.0)

        return p2d, p3d, normal, tex_coord

    def interpolate_triangle(self, P1: vector3.Vector3, P2: vector3.Vector3, P3: vector3.Vector3, N1: vector3.Vector3, N2: vector3.Vector3, N3: vector3.Vector3, M3D: vector3.Vector3) -> vector3.Vector3:
        N = vector3.Vector3()
        eps = 0.0001

        if ((P3.y - M3D.y) * (P2.y - M3D.y)) >= 0:
            if abs(P2.y - P1.y) > eps:
                rap1 = (M3D.y - P1.y) / (P2.y - P1.y)
                xa = P1.x + (P2.x - P1.x) * rap1
                Na = N1 + (N2 - N1) * rap1
                if abs(P3.y - P1.y) > eps:
                    rap2 = (M3D.y - P1.y) / (P3.y - P1.y)
                    xb = P1.x + (P3.x - P1.x) * rap2
                    Nb = N1 + (N3 - N1) * rap2
                    if abs(xa - xb) > eps:
                        N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                    else:
                        N = N1
                else:
                    N = N1
            else:
                N = N1
        else:
            if ((P1.y - M3D.y) * (P2.y - M3D.y)) >= 0:
                if abs(P2.y - P3.y) > eps:
                    rap1 = (M3D.y - P3.y) / (P2.y - P3.y)
                    xa = P3.x + (P2.x - P3.x) * rap1
                    Na = N3 + (N2 - N3) * rap1
                    if abs(P3.y - P1.y) > eps:
                        rap2 = (M3D.y - P3.y) / (P1.y - P3.y)
                        xb = P3.x + (P1.x - P3.x) * rap2
                        Nb = N3 + (N1 - N3) * rap2
                        if abs(xa - xb) > eps:
                            N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1
            else:
                if abs(P1.y - P2.y) > eps:
                    rap1 = (M3D.y - P2.y) / (P1.y - P2.y)
                    xa = P2.x + (P1.x - P2.x) * rap1
                    Na = N2 + (N1 - N2) * rap1
                    if abs(P2.y - P3.y) > eps:
                        rap2 = (M3D.y - P2.y) / (P3.y - P2.y)
                        xb = P2.x + (P3.x - P2.x) * rap2
                        Nb = N2 + (N3 - N2) * rap2
                        if abs(xa - xb) > eps:
                            N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1

        return N.normalize() if N.norm() > 0.00001 else N1

    def interpolate_triangle_textured(self, P1: vector3.Vector3, P2: vector3.Vector3, P3: vector3.Vector3, N1: vector3.Vector3, N2: vector3.Vector3, N3: vector3.Vector3, T1: Tuple[float, float], T2: Tuple[float, float], T3: Tuple[float, float], M3D: vector3.Vector3) -> Tuple[vector3.Vector3, Tuple[float, float]]:
        N = vector3.Vector3()
        T = T1
        eps = 0.0001

        if ((P3.y - M3D.y) * (P2.y - M3D.y)) >= 0:
            if abs(P2.y - P1.y) > eps:
                rap1 = (M3D.y - P1.y) / (P2.y - P1.y)
                xa = P1.x + (P2.x - P1.x) * rap1
                Na = N1 + (N2 - N1) * rap1
                Ta_x = T1[0] + rap1 * (T2[0] - T1[0])
                Ta_y = T1[1] + rap1 * (T2[1] - T1[1])
                if abs(P3.y - P1.y) > eps:
                    rap2 = (M3D.y - P1.y) / (P3.y - P1.y)
                    xb = P1.x + (P3.x - P1.x) * rap2
                    Nb = N1 + (N3 - N1) * rap2
                    Tb_x = T1[0] + rap2 * (T3[0] - T1[0])
                    Tb_y = T1[1] + rap2 * (T3[1] - T1[1])
                    if abs(xa - xb) > eps:
                        N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                        T = (Tb_x + ((M3D.x - xb) / (xa - xb)) * (Ta_x - Tb_x),
                             Tb_y + ((M3D.x - xb) / (xa - xb)) * (Ta_y - Tb_y))
                    else:
                        N = N1
                else:
                    N = N1
            else:
                N = N1
        else:
            if ((P1.y - M3D.y) * (P2.y - M3D.y)) >= 0:
                if abs(P2.y - P3.y) > eps:
                    rap1 = (M3D.y - P3.y) / (P2.y - P3.y)
                    xa = P3.x + (P2.x - P3.x) * rap1
                    Na = N3 + (N2 - N3) * rap1
                    Ta_x = T3[0] + rap1 * (T2[0] - T3[0])
                    Ta_y = T3[1] + rap1 * (T2[1] - T3[1])
                    if abs(P3.y - P1.y) > eps:
                        rap2 = (M3D.y - P3.y) / (P1.y - P3.y)
                        xb = P3.x + (P1.x - P3.x) * rap2
                        Nb = N3 + (N1 - N3) * rap2
                        Tb_x = T3[0] + rap2 * (T1[0] - T3[0])
                        Tb_y = T3[1] + rap2 * (T1[1] - T3[1])
                        if abs(xa - xb) > eps:
                            N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                            T = (Tb_x + ((M3D.x - xb) / (xa - xb)) * (Ta_x - Tb_x),
                                 Tb_y + ((M3D.x - xb) / (xa - xb)) * (Ta_y - Tb_y))
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1
            else:
                if abs(P1.y - P2.y) > eps:
                    rap1 = (M3D.y - P2.y) / (P1.y - P2.y)
                    xa = P2.x + (P1.x - P2.x) * rap1
                    Na = N2 + (N1 - N2) * rap1
                    Ta_x = T2[0] + rap1 * (T1[0] - T2[0])
                    Ta_y = T2[1] + rap1 * (T1[1] - T2[1])
                    if abs(P2.y - P3.y) > eps:
                        rap2 = (M3D.y - P2.y) / (P3.y - P2.y)
                        xb = P2.x + (P3.x - P2.x) * rap2
                        Nb = N2 + (N3 - N2) * rap2
                        Tb_x = T2[0] + rap2 * (T3[0] - T2[0])
                        Tb_y = T2[1] + rap2 * (T3[1] - T2[1])
                        if abs(xa - xb) > eps:
                            N = Nb + (Na - Nb) * ((M3D.x - xb) / (xa - xb))
                            T = (Tb_x + ((M3D.x - xb) / (xa - xb)) * (Ta_x - Tb_x),
                                 Tb_y + ((M3D.x - xb) / (xa - xb)) * (Ta_y - Tb_y))
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1

        if N.norm() > 0.00001:
            N = N.normalize()
        else:
            N = N1

        return N, T

    def calculate_color(self, M3D: Point3D, scene: SceneData) -> Couleur:
        _, p1_3d, n1_vec, _ = self._get_vertex_data(0)
        _, p2_3d, n2_vec, _ = self._get_vertex_data(1)
        _, p3_3d, n3_vec, _ = self._get_vertex_data(2)

        n1_vec = n1_vec.normalize()
        n2_vec = n2_vec.normalize()
        n3_vec = n3_vec.normalize()

        N = vector3.Vector3()
        T_coords = (0.0, 0.0)
        M3D_vec = vector3.Vector3(M3D[0], M3D[1], M3D[2])

        if self.face_properties.texture_on:
            _, _, _, T1 = self._get_vertex_data(0)
            _, _, _, T2 = self._get_vertex_data(1)
            _, _, _, T3 = self._get_vertex_data(2)
            
            N, T_coords = self.interpolate_triangle_textured(p1_3d, p2_3d, p3_3d, n1_vec, n2_vec, n3_vec, T1, T2, T3, M3D_vec)

            tex_u = max(0.0, min(1.0, T_coords[0]))
            tex_v = max(0.0, min(1.0, T_coords[1]))

            obj = self.scene.objects[self.face_properties.object_index]
            tex_width, tex_height = obj.texture_size[0]

            i = max(0, min(tex_width - 1, int((tex_width - 1) * tex_u)))
            j = max(0, min(tex_height - 1, int((tex_height - 1) * tex_v)))

            ind = j * tex_width + i
            cr, cv, cb = obj.texture_image[ind][:3]
        else:
            N = self.interpolate_triangle(p1_3d, p2_3d, p3_3d, n1_vec, n2_vec, n3_vec, M3D_vec)
            cr, cv, cb = self.face_properties.color

        V = (-M3D_vec).normalize() # Camera at origin
        R = (2 * (N.dot_product(V)) * N - V).normalize()

        coulR = self.scene.ambient_intensity * self.face_properties.coeffs[0] * cr
        coulG = self.scene.ambient_intensity * self.face_properties.coeffs[0] * cv
        coulB = self.scene.ambient_intensity * self.face_properties.coeffs[0] * cb

        for lum in self.scene.lights:
            L = (vector3.Vector3(lum[0], lum[1], lum[2]) - M3D_vec).normalize()
            costheta = L.dot_product(N)
            if costheta > 0:
                val = lum[3] * self.face_properties.coeffs[1] * costheta
                coulR += val * cr
                coulG += val * cv
                coulB += val * cb

            cosalpha = L.dot_product(R)
            if cosalpha > 0:
                val = lum[3] * self.face_properties.coeffs[2] * (cosalpha ** self.face_properties.coeffs[3]) * 255
                coulR += val
                coulG += val
                coulB += val

        return (min(255, int(coulR)), min(255, int(coulG)), min(255, int(coulB)))

    def fill(self, draw_point: DrawPointCallable, zbuffer: ZBuffer, scene: SceneData) -> None:
        p0_2d, _, _, _ = self._get_vertex_data(0)
        p1_2d, _, _, _ = self._get_vertex_data(1)
        p2_2d, _, _, _ = self._get_vertex_data(2)

        p_indices_sorted = self.sort_by_y(p0_2d[1], p1_2d[1], p2_2d[1])

        p_min_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[0]] - 1]
        p_moy_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[1]] - 1]
        p_max_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[2]] - 1]

        v0_3d = self.transformed_vertices_3d[self.vertex_indices[0] - 1]
        v1_3d = self.transformed_vertices_3d[self.vertex_indices[1] - 1]
        v2_3d = self.transformed_vertices_3d[self.vertex_indices[2] - 1]

        vec1 = v1_3d - v0_3d
        vec2 = v2_3d - v0_3d
        cross_product = vec1.cross_product(vec2)
        
        A, B, C = cross_product.x, cross_product.y, cross_product.z
        D = -(A * v0_3d.x + B * v0_3d.y + C * v0_3d.z)

        if C == 0: return # Parallel to camera plane

        self.face_properties.normal_and_plane = [A, B, C, D]
        
        Pmin_x, Pmin_y = round(p_min_2d[0]), round(p_min_2d[1])
        Pmoy_x, Pmoy_y = round(p_moy_2d[0]), round(p_moy_2d[1])
        Pmax_x, Pmax_y = round(p_max_2d[0]), round(p_max_2d[1])

        d = self.scene.camera_distance
        dy, dx = Pmax_y - Pmin_y, Pmax_x - Pmin_x

        if ((dy * (Pmoy_x - Pmin_x) - dx * (Pmoy_y - Pmin_y)) < 0):
            # Pmoy to the left (config 1)
            if (Pmin_y == Pmoy_y):
                if Pmax_y - Pmoy_y == 0: return
                edgeG = edge.Edge(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, 0 if (Pmax_x - Pmoy_x) <= 0 else (Pmax_y - Pmoy_y) - 1)
                edgeD = edge.Edge(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, (Pmin_y - Pmax_y) if (Pmax_x - Pmin_x) <= 0 else -1)
            else:
                if Pmoy_y - Pmin_y == 0: return
                edgeG = edge.Edge(Pmoy_y, Pmin_x, Pmoy_x - Pmin_x, Pmoy_y - Pmin_y, 0 if (Pmoy_x - Pmin_x) <= 0 else (Pmoy_y - Pmin_y) - 1)
                if Pmax_y - Pmin_y == 0: return
                edgeD = edge.Edge(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, (Pmin_y - Pmax_y) if (Pmax_x - Pmin_x) <= 0 else -1)

            y = Pmin_y
            while (y < Pmoy_y):
                for x in range(edgeG.x, edgeD.x + 1):
                    x_cam, y_cam = x - self.width // 2, (self.height + 1) // 2 - 1 - y
                    if (A * x_cam + B * y_cam + C * d) == 0: continue
                    t = -D / (A * x_cam + B * y_cam + C * d)
                    M3D = (t * x_cam, t * y_cam, t * d)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.zbuffer is not None:
                            if (M3D[2] > 0 and M3D[2] < self.zbuffer.get(x, y)):
                                self.zbuffer.set(x, y, M3D[2])
                                draw_point((x, y), self.calculate_color(M3D, self.scene))
                        else:
                            draw_point((x, y), self.calculate_color(M3D, self.scene))
                edgeG.update(); edgeD.update(); y += 1
            
            if Pmax_y - Pmoy_y == 0: return
            edgeG = edge.Edge(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, 0 if (Pmax_x - Pmoy_x) <= 0 else (Pmax_y - Pmoy_y) - 1)
            y = Pmoy_y
            while (y < Pmax_y):
                for x in range(edgeG.x, edgeD.x + 1):
                    x_cam, y_cam = x - self.width // 2, (self.height + 1) // 2 - 1 - y
                    if (A * x_cam + B * y_cam + C * d) == 0: continue
                    t = -D / (A * x_cam + B * y_cam + C * d)
                    M3D = (t * x_cam, t * y_cam, t * d)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.zbuffer is not None:
                            if (M3D[2] > 0 and M3D[2] < self.zbuffer.get(x, y)):
                                self.zbuffer.set(x, y, M3D[2])
                                draw_point((x, y), self.calculate_color(M3D, self.scene))
                        else:
                            draw_point((x, y), self.calculate_color(M3D, self.scene))
                edgeG.update(); edgeD.update(); y += 1
        else:
            # Pmoy to the right (config 2)
            if (Pmin_y == Pmoy_y):
                if Pmax_y - Pmin_y == 0: return
                edgeG = edge.Edge(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, 0 if (Pmax_x - Pmin_x) <= 0 else (Pmax_y - Pmin_y) - 1)
                if Pmax_y - Pmoy_y == 0: return
                edgeD = edge.Edge(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, (Pmoy_y - Pmax_y) if (Pmax_x - Pmoy_x) <= 0 else -1)
            else:
                if Pmax_y - Pmin_y == 0: return
                edgeG = edge.Edge(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, 0 if (Pmax_x - Pmin_x) <= 0 else (Pmax_y - Pmin_y) - 1)
                if Pmoy_y - Pmin_y == 0: return
                edgeD = edge.Edge(Pmoy_y, Pmin_x, Pmoy_x - Pmin_x, Pmoy_y - Pmin_y, (Pmin_y - Pmoy_y) if (Pmoy_x - Pmin_x) <= 0 else -1)

            y = Pmin_y
            while (y < Pmoy_y):
                for x in range(edgeG.x, edgeD.x + 1):
                    x_cam, y_cam = x - self.width // 2, (self.height + 1) // 2 - 1 - y
                    if (A * x_cam + B * y_cam + C * d) == 0: continue
                    t = -D / (A * x_cam + B * y_cam + C * d)
                    M3D = (t * x_cam, t * y_cam, t * d)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.zbuffer is not None:
                            if (M3D[2] > 0 and M3D[2] < self.zbuffer.get(x, y)):
                                self.zbuffer.set(x, y, M3D[2])
                                draw_point((x, y), self.calculate_color(M3D, self.scene))
                        else:
                            draw_point((x, y), self.calculate_color(M3D, self.scene))
                edgeG.update(); edgeD.update(); y += 1
            
            if Pmax_y - Pmoy_y == 0: return
            edgeD = edge.Edge(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, (Pmoy_y - Pmax_y) if (Pmax_x - Pmoy_x) <= 0 else -1)
            y = Pmoy_y
            while (y < Pmax_y):
                for x in range(edgeG.x, edgeD.x + 1):
                    x_cam, y_cam = x - self.width // 2, (self.height + 1) // 2 - 1 - y
                    if (A * x_cam + B * y_cam + C * d) == 0: continue
                    t = -D / (A * x_cam + B * y_cam + C * d)
                    M3D = (t * x_cam, t * y_cam, t * d)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.zbuffer is not None:
                            if (M3D[2] > 0 and M3D[2] < self.zbuffer.get(x, y)):
                                self.zbuffer.set(x, y, M3D[2])
                                draw_point((x, y), self.calculate_color(M3D, self.scene))
                        else:
                            draw_point((x, y), self.calculate_color(M3D, self.scene))
                edgeG.update(); edgeD.update(); y += 1

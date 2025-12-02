from __future__ import annotations
from typing import List, Tuple, Callable, Any

from . import AArete
from . import vecteur3
# This import is just for type annotation to avoid circular dependencies at runtime
from .Import_scene import Donnees_scene, Polyedre

# Type aliases for clarity
Point2D = Tuple[int, int]
Point3D = Tuple[float, float, float]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]


class Courbe(object):
    """ Classe generique definissant une courbe. """

    def __init__(self) -> None:
        self.controles: List[Point2D] = []

    def dessinerControles(self, dessinerControle: DrawControlCallable) -> None:
        """ Dessine les points de controle de la courbe. """
        for controle in self.controles:
            dessinerControle(controle)

    def dessinerPoints(self, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine la courbe. Methode a redefinir dans les classes derivees. """
        pass

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle. """
        # print point
        self.controles.append(point)

    def remplir(self, dessinerPoint: DrawPointCallable, *args: Any) -> None:
        """ remplir une courbe fermee : triangle"""
        pass


class Horizontale(Courbe):
    """ Definit une horizontale. Derive de Courbe. """

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle a l'horizontale.
        Ne fait rien si les 2 points existent deja. """
        if len(self.controles) < 2:
            super().ajouterControle(point)

    def dessinerPoints(self, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine la courbe. Redefinit la methode de la classe mere. """
        if len(self.controles) == 2:
            x1 = self.controles[0][0]
            x2 = self.controles[1][0]
            y = self.controles[0][1]
            xMin = min(x1, x2)
            xMax = max(x1, x2)
            for x in range(xMin, xMax):
                dessinerPoint((x, y), (0, 0, 0))


class Verticale(Courbe):
    """ Definit une verticale. Derive de Courbe. """

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle a la verticale.
        Ne fait rien si les 2 points existent deja. """
        if len(self.controles) < 2:
            super().ajouterControle(point)

    def dessinerPoints(self, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine la courbe. Redefinit la methode de la classe mere. """
        if len(self.controles) == 2:
            x = self.controles[0][0]
            y1 = self.controles[0][1]
            y2 = self.controles[1][1]
            yMin = min(y1, y2)
            yMax = max(y1, y2)
            for y in range(yMin, yMax):
                dessinerPoint((x, y), (0, 200, 200))


class GaucheDroite(Courbe):
    """ Definit une verticale. Derive de Courbe. """

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle a la verticale.
        Ne fait rien si les 2 points existent deja. """
        if len(self.controles) < 2:
            super().ajouterControle(point)

    def dessinerPoints(self, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine la courbe. Redefinit la methode de la classe mere. """
        if len(self.controles) == 2:
            x1 = self.controles[0][0]
            x2 = self.controles[1][0]
            y1 = self.controles[0][1]
            y2 = self.controles[1][1]

            if (y1 > y2):
                xint = x1
                x1 = x2
                x2 = xint
                yint = y1
                y1 = y2
                y2 = yint

            num = x2 - x1  # arete supposee gauche
            den = y2 - y1
            if den == 0: return
            inc = 0
            x = x1
            y = y1

            if (num > 0):
                inc = den - 1

            while (y < y2):
                dessinerPoint((x, y), (0, 255, 0))
                inc += num
                Q = inc // den
                x += Q
                inc -= Q * den
                y += 1

            # arete supposee droite

            inc = -den
            x = x1
            y = y1

            if (num > 0):
                inc = -1

            while (y < y2):
                dessinerPoint((x, y), (255, 0, 0))
                inc += num
                Q = inc // den
                x += Q
                inc -= Q * den
                y += 1


class DroiteMilieu(Courbe):
    """ Definit un segment algo point milieu. Derive de Courbe. """

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle a la droite.
        Ne fait rien si les 2 points existent deja. """
        if len(self.controles) < 2:
            super().ajouterControle(point)

    def dessinerPoints(self, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine la courbe. Redefinit la methode de la classe mere. """
        if len(self.controles) == 2:
            x1 = self.controles[0][0]
            x2 = self.controles[1][0]
            y1 = self.controles[0][1]
            y2 = self.controles[1][1]

            if y1 > y2:  # permutation des points
                y = y1
                x = x1
                y1 = y2
                x1 = x2
                y2 = y
                x2 = x
            dy = y2 - y1  # dy>0

            if (x2 >= x1):
                dx = x2 - x1
                y = y1
                if (dx >= dy):
                    dp = 2 * dy - dx
                    deltaE = 2 * dy
                    deltaNE = 2 * (dy - dx)
                    dessinerPoint((x1, y1), (0, 0, 0))
                    for x in range(x1, x2):
                        if (dp <= 0):
                            dp += deltaE
                        else:
                            dp += deltaNE
                            y += 1
                        dessinerPoint((x, y), (0, 0, 0))
                else:
                    dp = 2 * dx - dy
                    deltaE = 2 * dx
                    deltaNE = 2 * (dx - dy)
                    dessinerPoint((x1, y1), (0, 0, 0))
                    x = x1
                    for y in range(y1, y2):
                        if (dp <= 0):
                            dp += deltaE
                        else:
                            dp += deltaNE
                            x += 1
                        dessinerPoint((x, y), (0, 0, 0))
            else:
                dx = x1 - x2
                y = y1
                if (dx >= dy):
                    dp = 2 * dy - dx
                    deltaE = 2 * dy
                    deltaNE = 2 * (dy - dx)
                    dessinerPoint((x1, y1), (0, 0, 0))
                    for x in range(x1, x2, -1):
                        if (dp <= 0):
                            dp += deltaE
                        else:
                            dp += deltaNE
                            y += 1
                        dessinerPoint((x, y), (0, 0, 0))
                else:
                    dp = 2 * dx - dy
                    deltaE = 2 * dx
                    deltaNE = 2 * (dx - dy)
                    dessinerPoint((x1, y1), (0, 0, 0))
                    x = x1
                    for y in range(y1, y2):
                        if (dp <= 0):
                            dp += deltaE
                        else:
                            dp += deltaNE
                            x -= 1
                        dessinerPoint((x, y), (0, 0, 0))





class ZBuffer():
    def __init__(self) -> None:
        self.zbuffer: List[List[float]] = []
        self.dimx: int = 0
        self.dimy: int = 0

    def alloc_init_zbuffer(self, larg: int, haut: int) -> None:
        """initialisation du z-buffer"""
        self.zbuffer = [[] for i in range(larg)]
        self.dimx = larg
        self.dimy = haut
        for i in range(larg):
            col = haut * [100000.0]
            self.zbuffer[i] = col

    def acces(self, i: int, j: int) -> float:
        return self.zbuffer[i][j]

    def modif(self, i: int, j: int, val: float) -> None:
        self.zbuffer[i][j] = val


class Facette():
    def __init__(self) -> None:
        self.normaleetplan: List[float] = []
        self.texture_on: bool = False
        self.couleur: Couleur = (0, 0, 0)
        self.coefs: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
        self.indiceobjet: int = -1

class RenderedTriangle(Courbe):
    def __init__(self, polyedre_ref: Polyedre, triangle_face_index: int,
                 transformed_vertices_3d: List[vecteur3.Vecteur],
                 projected_vertices_2d: List[Point2D],
                 scene_ref: Donnees_scene,
                 zbuffer_ref: ZBuffer) -> None:
        super().__init__()
        self.polyedre = polyedre_ref
        self.triangle_face_index = triangle_face_index
        self.transformed_vertices_3d = transformed_vertices_3d
        self.projected_vertices_2d = projected_vertices_2d
        self.scene = scene_ref
        self.zbuffer = zbuffer_ref

        # Get the actual face data from the Polyedre using triangle_face_index
        self.vertex_indices = self.polyedre.listeindicestriangle[self.triangle_face_index]
        self.normal_indices = self.polyedre.listeindicesnormales[self.triangle_face_index]
        self.texture_indices = self.polyedre.listeindicestextures[self.triangle_face_index]
        self.face_color = self.polyedre.listecouleurs[self.triangle_face_index]
        self.face_coefs = self.polyedre.listecoefs[self.triangle_face_index]
        self.face_texture_on = self.polyedre.texture_on # Texture status for the whole object
        
        # Facette object to hold material/lighting properties, now stripped of duplicated vertex data
        self.facette_properties = Facette()
        self.facette_properties.couleur = self.face_color
        self.facette_properties.coefs = self.face_coefs
        self.facette_properties.texture_on = self.face_texture_on
        self.facette_properties.indiceobjet = self.polyedre.indice_objet

    def tri(self, y1: float, y2: float, y3: float) -> List[int]:
        if (y1 <= y2):
            if (y2 <= y3):
                i = 0
                j = 1
                k = 2
            else:
                if (y1 <= y3):
                    i = 0
                    j = 2
                    k = 1
                else:
                    i = 2
                    j = 0
                    k = 1
        else:  # y1>y2
            if (y2 > y3):
                i = 2
                j = 1
                k = 0

            else:
                if (y3 > y1):
                    i = 1
                    j = 0
                    k = 2

                else:
                    i = 1
                    j = 2
                    k = 0
        return [i, j, k]

    def _get_vertex_data(self, local_vertex_index: int) -> Tuple[Point2D, vecteur3.Vecteur, vecteur3.Vecteur, Tuple[float, float]]:
        # Get 1-based indices from the face definition
        v_idx_1based = self.vertex_indices[local_vertex_index]
        n_idx_1based = self.normal_indices[local_vertex_index] if self.normal_indices else 0 # Default to 0 or handle no normals
        t_idx_1based = self.texture_indices[local_vertex_index] if self.texture_indices else 0 # Default to 0 or handle no textures

        # Convert to 0-based indices for list access
        v_idx_0based = v_idx_1based - 1
        n_idx_0based = n_idx_1based - 1
        t_idx_0based = t_idx_1based - 1

        # Get 2D projected point
        p2d = self.projected_vertices_2d[v_idx_0based]

        # Get 3D transformed vertex (in camera space)
        p3d = self.transformed_vertices_3d[v_idx_0based]

        # Get normal (in camera space)
        # Use try-except for robustness if normal_indices is empty or invalid
        normal_data = self.polyedre.listenormales[n_idx_0based] if self.polyedre.listenormales and 0 <= n_idx_0based < len(self.polyedre.listenormales) else [0.0, 0.0, 0.0]
        normal = vecteur3.Vecteur(normal_data[0], normal_data[1], normal_data[2])

        # Get texture coordinate
        # Use try-except for robustness if texture_indices is empty or invalid
        tex_coord = self.polyedre.listecoordtextures[t_idx_0based] if self.polyedre.listecoordtextures and 0 <= t_idx_0based < len(self.polyedre.listecoordtextures) else (0.0, 0.0)

        return p2d, p3d, normal, tex_coord

    def interpoltriangle(self, P1: vecteur3.Vecteur, P2: vecteur3.Vecteur, P3: vecteur3.Vecteur, N1: vecteur3.Vecteur, N2: vecteur3.Vecteur, N3: vecteur3.Vecteur, M3D: vecteur3.Vecteur) -> vecteur3.Vecteur:
        # This method is directly copied from TriangleRempliZBuffer, as it's purely interpolation logic
        N = vecteur3.Vecteur()
        Na = vecteur3.Vecteur()
        Nb = vecteur3.Vecteur()
        eps = 0.0001

        if ((P3.y - M3D.y) * (P2.y - M3D.y)) >= 0:  # P3 et P2 du meme cote
            if abs(P2.y - P1.y) > eps:
                rap1 = (M3D.y - P1.y) / (P2.y - P1.y)
                xa = P1.x + (P2.x - P1.x) * rap1
                Na = N1 + rap1 * (N2 - N1)
                if abs(P3.y - P1.y) > eps:
                    rap2 = (M3D.y - P1.y) / (P3.y - P1.y)
                    xb = P1.x + (P3.x - P1.x) * rap2
                    Nb = N1 + rap2 * (N3 - N1)
                    if abs(xa - xb) > eps:
                        N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
                    else:
                        N = N1
                else:
                    N = N1
            else:
                N = N1

        else:
            if ((P1.y - M3D.y) * (P2.y - M3D.y)) >= 0:  # P1 et P2 du meme cote
                if abs(P2.y - P3.y) > eps:
                    rap1 = (M3D.y - P3.y) / (P2.y - P3.y)
                    xa = P3.x + (P2.x - P3.x) * rap1
                    Na = N3 + rap1 * (N2 - N3)
                    if abs(P3.y - P1.y) > eps:
                        rap2 = (M3D.y - P3.y) / (P1.y - P3.y)
                        xb = P3.x + (P1.x - P3.x) * rap2
                        Nb = N3 + rap2 * (N1 - N3)
                        if abs(xa - xb) > eps:
                            N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
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
                    Na = N2 + rap1 * (N1 - N2)
                    if abs(P2.y - P3.y) > eps:
                        rap2 = (M3D.y - P2.y) / (P3.y - P2.y)
                        xb = P2.x + (P3.x - P2.x) * rap2
                        Nb = N2 + rap2 * (N3 - N2)
                        if abs(xa - xb) > eps:
                            N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1

        if N.norm() > 0.00001:
            N = N.normer()
        else:
            N = N1

        return N

    def interpoltriangle1(self, P1: vecteur3.Vecteur, P2: vecteur3.Vecteur, P3: vecteur3.Vecteur, N1: vecteur3.Vecteur, N2: vecteur3.Vecteur, N3: vecteur3.Vecteur, T1: Tuple[float, float], T2: Tuple[float, float], T3: Tuple[float, float], M3D: vecteur3.Vecteur) -> List:
        # This method is directly copied from TriangleRempliZBuffer, as it's purely interpolation logic
        N = vecteur3.Vecteur()
        Na = vecteur3.Vecteur()
        Nb = vecteur3.Vecteur()
        eps = 0.0001

        T = T1

        if ((P3.y - M3D.y) * (P2.y - M3D.y)) >= 0:  # P3 et P2 du meme cote
            if abs(P2.y - P1.y) > eps:
                rap1 = (M3D.y - P1.y) / (P2.y - P1.y)
                xa = P1.x + (P2.x - P1.x) * rap1
                Na = N1 + rap1 * (N2 - N1)
                Ta_x = T1[0] + rap1 * (T2[0] - T1[0])
                Ta_y = T1[1] + rap1 * (T2[1] - T1[1])
                if abs(P3.y - P1.y) > eps:
                    rap2 = (M3D.y - P1.y) / (P3.y - P1.y)
                    xb = P1.x + (P3.x - P1.x) * rap2
                    Nb = N1 + rap2 * (N3 - N1)
                    Tb_x = T1[0] + rap2 * (T3[0] - T1[0])
                    Tb_y = T1[1] + rap2 * (T3[1] - T1[1])
                    if abs(xa - xb) > eps:
                        N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
                        T = (Tb_x + ((M3D.x - xb) / (xa - xb)) * (Ta_x - Tb_x),
                             Tb_y + ((M3D.x - xb) / (xa - xb)) * (Ta_y - Tb_y))
                    else:
                        N = N1
                else:
                    N = N1
            else:
                N = N1


        else:
            if ((P1.y - M3D.y) * (P2.y - M3D.y)) >= 0:  # P1 et P2 du meme cote
                if abs(P2.y - P3.y) > eps:
                    rap1 = (M3D.y - P3.y) / (P2.y - P3.y)
                    xa = P3.x + (P2.x - P3.x) * rap1
                    Na = N3 + rap1 * (N2 - N3)
                    Ta_x = T3[0] + rap1 * (T2[0] - T3[0])
                    Ta_y = T3[1] + rap1 * (T2[1] - T3[1])
                    if abs(P3.y - P1.y) > eps:
                        rap2 = (M3D.y - P3.y) / (P1.y - P3.y)
                        xb = P3.x + (P1.x - P3.x) * rap2
                        Nb = N3 + rap2 * (N1 - N3)
                        Tb_x = T3[0] + rap2 * (T1[0] - T3[0])
                        Tb_y = T3[1] + rap2 * (T1[1] - T3[1])
                        if abs(xa - xb) > eps:
                            N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
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
                    Na = N2 + rap1 * (N1 - N2)
                    Ta_x = T2[0] + rap1 * (T1[0] - T2[0])
                    Ta_y = T2[1] + rap1 * (T1[1] - T2[1])
                    if abs(P2.y - P3.y) > eps:
                        rap2 = (M3D.y - P2.y) / (P3.y - P2.y)
                        xb = P2.x + (P3.x - P2.x) * rap2
                        Nb = N2 + rap2 * (N3 - N2)
                        Tb_x = T2[0] + rap2 * (T3[0] - T2[0])
                        Tb_y = T2[1] + rap2 * (T3[1] - T2[1])
                        if abs(xa - xb) > eps:
                            N = Nb + ((M3D.x - xb) / (xa - xb)) * (Na - Nb)
                            T = (Tb_x + ((M3D.x - xb) / (xa - xb)) * (Ta_x - Tb_x),
                                 Tb_y + ((M3D.x - xb) / (xa - xb)) * (Ta_y - Tb_y))
                        else:
                            N = N1
                    else:
                        N = N1
                else:
                    N = N1

        if N.norm() > 0.00001:
            N = N.normer()
        else:
            N = N1

        return [N, T]

    def calculercouleur(self, M3D: Point3D, scene: Donnees_scene) -> Couleur:  # modele d illumination au point M3D
        # Get vertex normals for interpolation
        _, p1_3d, n1_vec, _ = self._get_vertex_data(0)
        _, p2_3d, n2_vec, _ = self._get_vertex_data(1)
        _, p3_3d, n3_vec, _ = self._get_vertex_data(2)

        # Normalize normals
        n1_vec = n1_vec.normer()
        n2_vec = n2_vec.normer()
        n3_vec = n3_vec.normer()

        N = vecteur3.Vecteur()
        T_coords = (0.0, 0.0) # Default texture coordinates

        # Convert M3D tuple to vecteur3.Vecteur once
        M3D_vec = vecteur3.Vecteur(M3D[0], M3D[1], M3D[2])

        if self.facette_properties.texture_on:
            _, _, _, T1 = self._get_vertex_data(0)
            _, _, _, T2 = self._get_vertex_data(1)
            _, _, _, T3 = self._get_vertex_data(2)
            
            [N, T_coords] = self.interpoltriangle1(p1_3d, p2_3d, p3_3d, n1_vec, n2_vec, n3_vec, T1, T2, T3, M3D_vec)

            # Texture mapping
            # Ensure T_coords are within [0, 1] for indexing
            tex_u = max(0.0, min(1.0, T_coords[0]))
            tex_v = max(0.0, min(1.0, T_coords[1]))

            # Adjust for potential 0 or 1 index if size is 1
            tex_width = self.scene.listeobjets[self.facette_properties.indiceobjet].texture_size[0][0]
            tex_height = self.scene.listeobjets[self.facette_properties.indiceobjet].texture_size[0][1]

            i = int((tex_width - 1) * tex_u)
            j = int((tex_height - 1) * tex_v)
            
            # Clamp indices to valid range
            i = max(0, min(tex_width - 1, i))
            j = max(0, min(tex_height - 1, j))

            ind = j * tex_width + i
            
            (cr, cv, cb) = (self.scene.listeobjets[self.facette_properties.indiceobjet].texture_ima[ind][0],
                           self.scene.listeobjets[self.facette_properties.indiceobjet].texture_ima[ind][1],
                           self.scene.listeobjets[self.facette_properties.indiceobjet].texture_ima[ind][2])
        else:
            N = self.interpoltriangle(p1_3d, p2_3d, p3_3d, n1_vec, n2_vec, n3_vec, M3D_vec)
            (cr, cv, cb) = self.facette_properties.couleur

        V = vecteur3.Vecteur(-M3D[0], -M3D[1], -M3D[2])  # camera en O
        V = V.normer()
        R = 2 * (N * V) * N - V  # les vecteurs sont normes

        coulR = self.scene.Ia * self.facette_properties.coefs[0] * cr
        coulG = self.scene.Ia * self.facette_properties.coefs[0] * cv
        coulB = self.scene.Ia * self.facette_properties.coefs[0] * cb

        for lum in self.scene.listelum:
            L = vecteur3.Vecteur(lum[0] - M3D[0], lum[1] - M3D[1], lum[2] - M3D[2])
            L = L.normer()
            costheta = L * N
            if costheta > 0:  # diffus
                val = lum[3] * self.facette_properties.coefs[1] * costheta
                coulR += val * cr
                coulG += val * cv
                coulB += val * cb

            cosalpha = L * R
            if cosalpha > 0:  # speculaire
                cosalphan = cosalpha
                i = 1
                while (i < self.facette_properties.coefs[3]):
                    cosalphan *= cosalpha
                    i += 1
                val = lum[3] * self.facette_properties.coefs[2] * cosalphan * 255
                coulR += val
                coulG += val
                coulB += val

        coulR = int(coulR)
        coulG = int(coulG)
        coulB = int(coulB)
        if coulR > 255:  # saturation
            coulR = 255
        if coulG > 255:
            coulG = 255
        if coulB > 255:
            coulB = 255

        return (coulR, coulG, coulB)

    def remplir(self, dessinerPoint: DrawPointCallable, zbuffer: ZBuffer, scene: Donnees_scene) -> None:
        # Get the 2D projected points for the current triangle
        p0_2d, _, _, _ = self._get_vertex_data(0)
        p1_2d, _, _, _ = self._get_vertex_data(1)
        p2_2d, _, _, _ = self._get_vertex_data(2)

        p_indices_sorted = self.tri(p0_2d[1], p1_2d[1], p2_2d[1])

        p_min_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[0]] - 1]
        p_moy_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[1]] - 1]
        p_max_2d = self.projected_vertices_2d[self.vertex_indices[p_indices_sorted[2]] - 1]

        # Get 3D transformed vertices (in camera space) for plane equation
        v0_3d = self.transformed_vertices_3d[self.vertex_indices[0] - 1]
        v1_3d = self.transformed_vertices_3d[self.vertex_indices[1] - 1]
        v2_3d = self.transformed_vertices_3d[self.vertex_indices[2] - 1]

        # Calculate plane equation (A*x + B*y + C*z + D = 0) in camera space
        # Vectors for plane normal calculation
        vec1 = v1_3d - v0_3d
        vec2 = v2_3d - v0_3d
        
        cross_product = vec1.produitVectoriel(vec2)
        
        A = cross_product.x
        B = cross_product.y
        C = cross_product.z
        D = -(A * v0_3d.x + B * v0_3d.y + C * v0_3d.z)

        self.facette_properties.normaleetplan = [A, B, C, D]
        # Rest of the scanline rasterization logic, adapted to use references
        # from projected_vertices_2d and transformed_vertices_3d
        
        Pmin_x = round(p_min_2d[0])
        Pmin_y = round(p_min_2d[1])
        Pmoy_x = round(p_moy_2d[0])
        Pmoy_y = round(p_moy_2d[1])
        Pmax_x = round(p_max_2d[0])
        Pmax_y = round(p_max_2d[1])

        d = self.scene.d

        dy = Pmax_y - Pmin_y
        dx = Pmax_x - Pmin_x

        if C == 0 : return # Avoid division by zero, parallel to camera plane

        # Determine if Pmoy is to the left or right of segment [Pmin, Pmax]
        if ((dy * (Pmoy_x - Pmin_x) - dx * (Pmoy_y - Pmin_y)) < 0):
            # Pmoy to the left (config 1)
            if (Pmin_y == Pmoy_y):  # Special case: horizontal bottom edge
                if Pmax_y - Pmoy_y == 0: return
                edgeG = AArete.Arete(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, 0 if (Pmax_x - Pmoy_x) <= 0 else (Pmax_y - Pmoy_y) - 1)
                edgeD = AArete.Arete(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, (Pmin_y - Pmax_y) if (Pmax_x - Pmin_x) <= 0 else -1)
            else: # General case config 1
                if Pmoy_y - Pmin_y == 0: return
                edgeG = AArete.Arete(Pmoy_y, Pmin_x, Pmoy_x - Pmin_x, Pmoy_y - Pmin_y, 0 if (Pmoy_x - Pmin_x) <= 0 else (Pmoy_y - Pmin_y) - 1)
                if Pmax_y - Pmin_y == 0: return
                edgeD = AArete.Arete(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, (Pmin_y - Pmax_y) if (Pmax_x - Pmin_x) <= 0 else -1)

            y = Pmin_y
            while (y < Pmoy_y):
                xG = edgeG.x
                xD = edgeD.x
                for x in range(xG, xD + 1):
                    # Calculate 3D point M3D using inverse perspective projection
                    if (A * x + B * y + C * d) == 0: continue # Avoid division by zero
                    t = -D / (A * x + B * y + C * d)
                    M3D = (t * x, t * y, t * d)
                    
                    posx = self.zbuffer.dimx // 2 + x
                    posy = (self.zbuffer.dimy + 1) // 2 - 1 - y

                    if 0 <= posx < self.zbuffer.dimx and 0 <= posy < self.zbuffer.dimy:
                        if (M3D[2] > 0 and M3D[2] < self.zbuffer.acces(posx, posy)):
                            self.zbuffer.modif(posx, posy, M3D[2])
                            coul = self.calculercouleur(M3D, self.scene)
                            dessinerPoint((posx, posy), coul)
                edgeG.maj()
                edgeD.maj()
                y += 1
            
            if Pmax_y - Pmoy_y == 0: return
            edgeG = AArete.Arete(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, 0 if (Pmax_x - Pmoy_x) <= 0 else (Pmax_y - Pmoy_y) - 1)
            y = Pmoy_y
            while (y < Pmax_y):
                xG = edgeG.x
                xD = edgeD.x
                for x in range(xG, xD + 1):
                    if (A * x + B * y + C * d) == 0: continue
                    t = -D / (A * x + B * y + C * d)
                    M3D = (t * x, t * y, t * d)
                    posx = self.zbuffer.dimx // 2 + x
                    posy = (self.zbuffer.dimy + 1) // 2 - 1 - y
                    if 0 <= posx < self.zbuffer.dimx and 0 <= posy < self.zbuffer.dimy:
                        if (M3D[2] > 0 and M3D[2] < self.zbuffer.acces(posx, posy)):
                            self.zbuffer.modif(posx, posy, M3D[2])
                            coul = self.calculercouleur(M3D, self.scene)
                            dessinerPoint((posx, posy), coul)
                edgeG.maj()
                edgeD.maj()
                y += 1

        else: # Pmoy to the right (config 2)
            if (Pmin_y == Pmoy_y): # Special case: horizontal bottom edge
                if Pmax_y - Pmin_y == 0: return
                edgeG = AArete.Arete(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, 0 if (Pmax_x - Pmin_x) <= 0 else (Pmax_y - Pmin_y) - 1)
                if Pmax_y - Pmoy_y == 0: return
                edgeD = AArete.Arete(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, (Pmoy_y - Pmax_y) if (Pmax_x - Pmoy_x) <= 0 else -1)
            else: # General case config 2
                if Pmax_y - Pmin_y == 0: return
                edgeG = AArete.Arete(Pmax_y, Pmin_x, Pmax_x - Pmin_x, Pmax_y - Pmin_y, 0 if (Pmax_x - Pmin_x) <= 0 else (Pmax_y - Pmin_y) - 1)
                if Pmoy_y - Pmin_y == 0: return
                edgeD = AArete.Arete(Pmoy_y, Pmin_x, Pmoy_x - Pmin_x, Pmoy_y - Pmin_y, (Pmin_y - Pmoy_y) if (Pmoy_x - Pmin_x) <= 0 else -1)

            y = Pmin_y
            while (y < Pmoy_y):
                xG = edgeG.x
                xD = edgeD.x
                for x in range(xG, xD + 1):
                    if (A * x + B * y + C * d) == 0: continue
                    t = -D / (A * x + B * y + C * d)
                    M3D = (t * x, t * y, t * d)
                    posx = self.zbuffer.dimx // 2 + x
                    posy = (self.zbuffer.dimy + 1) // 2 - 1 - y
                    if 0 <= posx < self.zbuffer.dimx and 0 <= posy < self.zbuffer.dimy:
                        if (M3D[2] > 0 and M3D[2] < self.zbuffer.acces(posx, posy)):
                            self.zbuffer.modif(posx, posy, M3D[2])
                            coul = self.calculercouleur(M3D, self.scene)
                            dessinerPoint((posx, posy), coul)
                edgeG.maj()
                edgeD.maj()
                y += 1
            
            if Pmax_y - Pmoy_y == 0: return
            edgeD = AArete.Arete(Pmax_y, Pmoy_x, Pmax_x - Pmoy_x, Pmax_y - Pmoy_y, (Pmoy_y - Pmax_y) if (Pmax_x - Pmoy_x) <= 0 else -1)
            y = Pmoy_y
            while (y < Pmax_y):
                xG = edgeG.x
                xD = edgeD.x
                for x in range(xG, xD + 1):
                    if (A * x + B * y + C * d) == 0: continue
                    t = -D / (A * x + B * y + C * d)
                    M3D = (t * x, t * y, t * d)
                    posx = self.zbuffer.dimx // 2 + x
                    posy = (self.zbuffer.dimy + 1) // 2 - 1 - y
                    if 0 <= posx < self.zbuffer.dimx and 0 <= posy < self.zbuffer.dimy:
                        if (M3D[2] > 0 and M3D[2] < self.zbuffer.acces(posx, posy)):
                            self.zbuffer.modif(posx, posy, M3D[2])
                            coul = self.calculercouleur(M3D, self.scene)
                            dessinerPoint((posx, posy), coul)
                edgeG.maj()
                edgeD.maj()
                y += 1
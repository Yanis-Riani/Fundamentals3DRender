from __future__ import annotations
from typing import List, Tuple, Callable, Any

from . import AArete
from . import vecteur3
# This import is just for type annotation to avoid circular dependencies at runtime
from .Import_scene import Donnees_scene

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


class TriangleRempli(Courbe):
    """ Remplit un triangle avec la couleur spécifiée au constructeur. """

    def __init__(self, couleur: Couleur) -> None:
        super().__init__()
        self.couleur = couleur

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de contrôle si moins de 3 points sont présents. """
        if len(self.controles) < 3:
            super().ajouterControle(point)

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

    def remplir(self, dessinerPoint: DrawPointCallable, *args: Any) -> None:

        N = len(self.controles)

        # construction du triangle comme cours avec ymin<=ymoy<=ymax
        if (N == 3):
            y1 = self.controles[0][1]
            y2 = self.controles[1][1]
            y3 = self.controles[2][1]

            ind = self.tri(y1, y2, y3)

            Pmin_x = round(self.controles[ind[0]][0])
            Pmin_y = round(self.controles[ind[0]][1])
            Pmoy_x = round(self.controles[ind[1]][0])
            Pmoy_y = round(self.controles[ind[1]][1])
            Pmax_x = round(self.controles[ind[2]][0])
            Pmax_y = round(self.controles[ind[2]][1])

            # 2 cas particuliers et 2 config generales
            dy = Pmax_y - Pmin_y
            dx = Pmax_x - Pmin_x

            if dy == 0: return

            if ((dy * (Pmoy_x - Pmin_x) - dx * (Pmoy_y - Pmin_y)) < 0):
                # Pmoy a gauche segment [Pmin,Pmax]
                # config 1 du cours
                if (Pmin_y == Pmoy_y):  # cas particulier
                    num = Pmax_x - Pmoy_x
                    den = Pmax_y-Pmoy_y
                    if den == 0: return
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, den - 1)

                    num = Pmax_x - Pmin_x
                    den = Pmax_y-Pmin_y
                    if den == 0: return
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, Pmin_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1
                else:
                    # cas general config 1
                    num = Pmoy_x - Pmin_x
                    den = Pmoy_y-Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmoy_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmoy_y, Pmin_x, num, den, den - 1)

                    num = Pmax_x - Pmin_x
                    den = Pmax_y-Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, Pmin_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmoy_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, den - 1)
                    y = Pmoy_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1


            else:
                # Pmoy a droite
                # config 2 du cours
                if (Pmin_y == Pmoy_y):  # cas particulier
                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: return
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, den - 1)

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: return
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, Pmoy_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                else:  # cas general config 2
                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, den - 1)

                    num = Pmoy_x - Pmin_x
                    den = Pmoy_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmoy_y, Pmin_x, num, den, Pmin_y - Pmoy_y)
                    else:
                        edgeD = AArete.Arete(Pmoy_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmoy_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)

                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, Pmoy_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, -1)
                    y = Pmoy_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            dessinerPoint((x, y), self.couleur)

                        edgeG.maj()
                        edgeD.maj()
                        y += 1


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
        self.sommets: List[Point3D] = []
        self.normaleetplan: List[float] = []
        self.normalesaux3sommets: List[Point3D] = []
        self.texture_on: bool = False
        self.coordtexturesaux3sommets: List[Tuple[float, float]] = []
        self.couleur: Couleur = (0, 0, 0)
        self.coefs: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
        self.indiceobjet: int = -1


class TriangleRempliZBuffer(Courbe):
    def __init__(self) -> None:
        super().__init__()
        self.facette: Facette = Facette()

    def ajouterControle(self, point: Point2D) -> None:
        """ Ajoute un point de controle au triangle.
        Ne fait rien si les 3 points existent deja. """
        if len(self.controles) < 3:
            super().ajouterControle(point)

    def ajouterfacette(self, facette: Facette) -> None:
        self.facette = facette

    def interpoltriangle(self, P1: Point3D, P2: Point3D, P3: Point3D, N1: vecteur3.Vecteur, N2: vecteur3.Vecteur, N3: vecteur3.Vecteur, M3D: Point3D) -> vecteur3.Vecteur:
        N = vecteur3.Vecteur()
        Na = vecteur3.Vecteur()
        Nb = vecteur3.Vecteur()
        eps = 0.0001

        if ((P3[1] - M3D[1]) * (P2[1] - M3D[1])) >= 0:  # P3 et P2 du meme cote
            if ((P2[1] - P1[1]) * (P2[1] - P1[1])) > eps:
                rap1 = (M3D[1] - P1[1]) / (P2[1] - P1[1])
                xa = P1[0] + (P2[0] - P1[0]) * rap1
                Na = N1 + rap1 * (N2 - N1)
                if ((P3[1] - P1[1]) * (P3[1] - P1[1])) > eps:
                    rap2 = (M3D[1] - P1[1]) / (P3[1] - P1[1])
                    xb = P1[0] + (P3[0] - P1[0]) * rap2
                    Nb = N1 + rap2 * (N3 - N1)
                    if (xa - xb) * (xa - xb) > eps:
                        N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
                    else:
                        N = N1
                else:
                    N = N1
            else:
                N = N1

        else:
            if ((P1[1] - M3D[1]) * (P2[1] - M3D[1])) >= 0:  # P1 et P2 du meme cote
                if ((P2[1] - P3[1]) * (P2[1] - P3[1])) > eps:
                    rap1 = (M3D[1] - P3[1]) / (P2[1] - P3[1])
                    xa = P3[0] + (P2[0] - P3[0]) * rap1
                    Na = N3 + rap1 * (N2 - N3)
                    if ((P3[1] - P1[1]) * (P3[1] - P1[1])) > eps:
                        rap2 = (M3D[1] - P3[1]) / (P1[1] - P3[1])
                        xb = P3[0] + (P1[0] - P3[0]) * rap2
                        Nb = N3 + rap2 * (N1 - N3)
                        if (xa - xb) * (xa - xb) > eps:
                            N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
                        else:
                            N = N1

                    else:
                        N = N1
                else:
                    N = N1


            else:
                if ((P1[1] - P2[1]) * (P1[1] - P2[1])) > eps:
                    rap1 = (M3D[1] - P2[1]) / (P1[1] - P2[1])
                    xa = P2[0] + (P1[0] - P2[0]) * rap1
                    Na = N2 + rap1 * (N1 - N2)
                    if ((P2[1] - P3[1]) * (P2[1] - P3[1])) > eps:
                        rap2 = (M3D[1] - P2[1]) / (P3[1] - P2[1])
                        xb = P2[0] + (P3[0] - P2[0]) * rap2
                        Nb = N2 + rap2 * (N3 - N2)
                        if (xa - xb) * (xa - xb) > eps:
                            N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
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

    def interpoltriangle1(self, P1: Point3D, P2: Point3D, P3: Point3D, N1: vecteur3.Vecteur, N2: vecteur3.Vecteur, N3: vecteur3.Vecteur, T1: Tuple[float, float], T2: Tuple[float, float], T3: Tuple[float, float], M3D: Point3D) -> List:
        N = vecteur3.Vecteur()
        Na = vecteur3.Vecteur()
        Nb = vecteur3.Vecteur()
        eps = 0.0001

        T = T1

        if ((P3[1] - M3D[1]) * (P2[1] - M3D[1])) >= 0:  # P3 et P2 du meme cote
            if ((P2[1] - P1[1]) * (P2[1] - P1[1])) > eps:
                rap1 = (M3D[1] - P1[1]) / (P2[1] - P1[1])
                xa = P1[0] + (P2[0] - P1[0]) * rap1
                Na = N1 + rap1 * (N2 - N1)
                Ta_x = T1[0] + rap1 * (T2[0] - T1[0])
                Ta_y = T1[1] + rap1 * (T2[1] - T1[1])
                if ((P3[1] - P1[1]) * (P3[1] - P1[1])) > eps:
                    rap2 = (M3D[1] - P1[1]) / (P3[1] - P1[1])
                    xb = P1[0] + (P3[0] - P1[0]) * rap2
                    Nb = N1 + rap2 * (N3 - N1)
                    Tb_x = T1[0] + rap2 * (T3[0] - T1[0])
                    Tb_y = T1[1] + rap2 * (T3[1] - T1[1])
                    if (xa - xb) * (xa - xb) > eps:
                        N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
                        T = (Tb_x + ((M3D[0] - xb) / (xa - xb)) * (Ta_x - Tb_x),
                             Tb_y + ((M3D[0] - xb) / (xa - xb)) * (Ta_y - Tb_y))
                        if T[0] < 0 or T[1] < 0 or T[0] > 1.0 or T[1] > 1.0:
                            # print ("T23 PB", T, Ta_x, " ", Ta_y, "  ", Tb_x, " ", Tb_y, " M3D= ", M3D, P1, P2, P3)
                            T = T1
                    else:
                        N = N1
                        T = T1
                else:
                    N = N1
                    T = T1
            else:
                N = N1
                T = T1


        else:
            if ((P1[1] - M3D[1]) * (P2[1] - M3D[1])) >= 0:  # P1 et P2 du meme cote
                if ((P2[1] - P3[1]) * (P2[1] - P3[1])) > eps:
                    rap1 = (M3D[1] - P3[1]) / (P2[1] - P3[1])
                    xa = P3[0] + (P2[0] - P3[0]) * rap1
                    Na = N3 + rap1 * (N2 - N3)
                    Ta_x = T3[0] + rap1 * (T2[0] - T3[0])
                    Ta_y = T3[1] + rap1 * (T2[1] - T3[1])
                    if ((P3[1] - P1[1]) * (P3[1] - P1[1])) > eps:
                        rap2 = (M3D[1] - P3[1]) / (P1[1] - P3[1])
                        xb = P3[0] + (P1[0] - P3[0]) * rap2
                        Nb = N3 + rap2 * (N1 - N3)
                        Tb_x = T3[0] + rap2 * (T1[0] - T3[0])
                        Tb_y = T3[1] + rap2 * (T1[1] - T3[1])
                        if (xa - xb) * (xa - xb) > eps:
                            N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
                            T = (Tb_x + ((M3D[0] - xb) / (xa - xb)) * (Ta_x - Tb_x),
                                 Tb_y + ((M3D[0] - xb) / (xa - xb)) * (Ta_y - Tb_y))
                            if T[0] < 0 or T[1] < 0 or T[0] > 1.0 or T[1] > 1.0:
                                # print ("T12 PB", T, Ta_x, " ", Ta_y, "  ", Tb_x, " ", Tb_y, " M3D= ", M3D, P1, P2, P3)
                                T = T1
                        else:
                            N = N1
                            T = T1
                    else:
                        N = N1
                        T = T1
                else:
                    N = N1
                    T = T1


            else:
                if ((P1[1] - P2[1]) * (P1[1] - P2[1])) > eps:
                    rap1 = (M3D[1] - P2[1]) / (P1[1] - P2[1])
                    xa = P2[0] + (P1[0] - P2[0]) * rap1
                    Na = N2 + rap1 * (N1 - N2)
                    Ta_x = T2[0] + rap1 * (T1[0] - T2[0])
                    Ta_y = T2[1] + rap1 * (T1[1] - T2[1])
                    if ((P2[1] - P3[1]) * (P2[1] - P3[1])) > eps:
                        rap2 = (M3D[1] - P2[1]) / (P3[1] - P2[1])
                        xb = P2[0] + (P3[0] - P2[0]) * rap2
                        Nb = N2 + rap2 * (N3 - N2)
                        Tb_x = T2[0] + rap2 * (T3[0] - T2[0])
                        Tb_y = T2[1] + rap2 * (T3[1] - T2[1])
                        if (xa - xb) * (xa - xb) > eps:
                            N = Nb + ((M3D[0] - xb) / (xa - xb)) * (Na - Nb)
                            T = (Tb_x + ((M3D[0] - xb) / (xa - xb)) * (Ta_x - Tb_x),
                                 Tb_y + ((M3D[0] - xb) / (xa - xb)) * (Ta_y - Tb_y))
                            if T[0] < 0 or T[1] < 0 or T[0] > 1.0 or T[1] > 1.0:
                                # print ("T13 PB", T, Ta_x, " ", Ta_y, "  ", Tb_x, " ", Tb_y, " M3D= ", M3D, P1, P2, P3)
                                T = T1
                        else:
                            N = N1
                            T = T1
                    else:
                        N = N1
                        T = T1
                else:
                    N = N1
                    T = T1

        if N.norm() > 0.00001:
            N = N.normer()
        else:
            N = N1

        return [N, T]

    def calculercouleur(self, M3D: Point3D, scene: Donnees_scene) -> Couleur:  # modele d illumination au point M3D
        N1 = vecteur3.Vecteur(self.facette.normalesaux3sommets[0][0], self.facette.normalesaux3sommets[0][1],
                               self.facette.normalesaux3sommets[0][2])
        N1 = N1.normer()
        N2 = vecteur3.Vecteur(self.facette.normalesaux3sommets[1][0], self.facette.normalesaux3sommets[1][1],
                               self.facette.normalesaux3sommets[1][2])
        N2 = N2.normer()
        N3 = vecteur3.Vecteur(self.facette.normalesaux3sommets[2][0], self.facette.normalesaux3sommets[2][1],
                               self.facette.normalesaux3sommets[2][2])
        N3 = N3.normer()

        P1 = self.facette.sommets[0]
        P2 = self.facette.sommets[1]
        P3 = self.facette.sommets[2]

        N = vecteur3.Vecteur()

        if self.facette.texture_on:

            T1 = (self.facette.coordtexturesaux3sommets[0][0], self.facette.coordtexturesaux3sommets[0][1])
            T2 = (self.facette.coordtexturesaux3sommets[1][0], self.facette.coordtexturesaux3sommets[1][1])
            T3 = (self.facette.coordtexturesaux3sommets[2][0], self.facette.coordtexturesaux3sommets[2][1])
            # print(M3D)

            [N, T] = self.interpoltriangle1(P1, P2, P3, N1, N2, N3, T1, T2, T3, M3D)
            # print (T1,T2,T3," et ",T)
            i = int((scene.listeobjets[self.facette.indiceobjet].texture_size[0][0] - 1) * T[0])
            j = int((scene.listeobjets[self.facette.indiceobjet].texture_size[0][1] - 1) * T[1])
            ind = j * scene.listeobjets[self.facette.indiceobjet].texture_size[0][0] + i
            # print(ind)
            (cr, cv, cb) = (scene.listeobjets[self.facette.indiceobjet].texture_ima[ind][0],
                           scene.listeobjets[self.facette.indiceobjet].texture_ima[ind][1],
                           scene.listeobjets[self.facette.indiceobjet].texture_ima[ind][2])

        else:
            N = self.interpoltriangle(P1, P2, P3, N1, N2, N3, M3D)

            # N=self.interpoltriangle(P1,P2,P3,N1,N2,N3,M3D)

            # couleur intrinseque de la facette
            (cr, cv, cb) = (self.facette.couleur[0], self.facette.couleur[1], self.facette.couleur[2])

        V = vecteur3.Vecteur(-M3D[0], -M3D[1], -M3D[2])  # camera en O
        V = V.normer()
        R = 2 * (N * V) * N - V  # les vecteurs sont normes

        coulR = scene.Ia * self.facette.coefs[0] * cr
        coulG = scene.Ia * self.facette.coefs[0] * cv
        coulB = scene.Ia * self.facette.coefs[0] * cb

        for lum in scene.listelum:
            L = vecteur3.Vecteur(lum[0] - M3D[0], lum[1] - M3D[1], lum[2] - M3D[2])
            L = L.normer()
            costheta = L * N
            if costheta > 0:  # diffus
                val = lum[3] * self.facette.coefs[1] * costheta
                coulR += val * cr
                coulG += val * cv
                coulB += val * cb

            cosalpha = L * R
            if cosalpha > 0:  # speculaire
                cosalphan = cosalpha
                i = 1
                while (i < self.facette.coefs[3]):
                    cosalphan *= cosalpha
                    i += 1
                val = lum[3] * self.facette.coefs[2] * cosalphan * 255
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

    def tri(self, y1: float, y2: float, y3: float) -> List[int]:
        """tri sur les valeurs de y1,y2,y3 en renvoyant les (indices -1) des elements :
        sert pour la determination de Pmin,Pmoy,Pmax"""
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

    def remplir(self, dessinerPoint: DrawPointCallable, zbuffer: ZBuffer, scene: Donnees_scene) -> None:

        N = len(self.controles)

        # construction du triangle comme cours avec ymin<=ymoy<=ymax
        if (N == 3):
            y1 = self.controles[0][1]
            y2 = self.controles[1][1]
            y3 = self.controles[2][1]

            ind = self.tri(y1, y2, y3)
            # trier les 3 points Pmin, Pmoy, Pmax
            Pmin_x = self.controles[ind[0]][0]
            Pmin_y = self.controles[ind[0]][1]
            Pmoy_x = self.controles[ind[1]][0]
            Pmoy_y = self.controles[ind[1]][1]
            Pmax_x = self.controles[ind[2]][0]
            Pmax_y = self.controles[ind[2]][1]

            # 2 cas particuliers et 2 config generales

            A = self.facette.normaleetplan[0]
            B = self.facette.normaleetplan[1]
            C = self.facette.normaleetplan[2]
            D = self.facette.normaleetplan[3]

            d = scene.d

            dy = Pmax_y - Pmin_y
            dx = Pmax_x - Pmin_x

            if C == 0 : return # Avoid division by zero

            if ((dy * (Pmoy_x - Pmin_x) - dx * (Pmoy_y - Pmin_y)) < 0):
                # Pmoy a gauche segment [Pmin,Pmax]
                # config 1 du cours
                if (Pmin_y == Pmoy_y):  # cas particulier
                    num = Pmax_x - Pmoy_x
                    den = Pmax_y-Pmoy_y
                    if den == 0: return
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, den - 1)

                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: return
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, Pmin_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y
                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1



                else:
                    # cas general config 1
                    num = Pmoy_x - Pmin_x
                    den = Pmoy_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmoy_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmoy_y, Pmin_x, num, den, den - 1)

                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, Pmin_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmoy_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y

                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmoy_x, num, den, den - 1)
                    y = Pmoy_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y

                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)

                        edgeG.maj()
                        edgeD.maj()
                        y += 1


            else:
                # Pmoy a droite
                # config 2 du cours
                if (Pmin_y == Pmoy_y):  # cas particulier
                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: return
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, den - 1)

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: return
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, Pmoy_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y

                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)
                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                else:  # cas general config 2
                    num = Pmax_x - Pmin_x
                    den = Pmax_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, 0)
                    else:
                        edgeG = AArete.Arete(Pmax_y, Pmin_x, num, den, den - 1)

                    num = Pmoy_x - Pmin_x
                    den = Pmoy_y - Pmin_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmoy_y, Pmin_x, num, den, Pmin_y - Pmoy_y)
                    else:
                        edgeD = AArete.Arete(Pmoy_y, Pmin_x, num, den, -1)

                    y = Pmin_y
                    while (y < Pmoy_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y

                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)

                        edgeG.maj()
                        edgeD.maj()
                        y += 1

                    num = Pmax_x - Pmoy_x
                    den = Pmax_y - Pmoy_y
                    if den == 0: den = 1
                    if (num <= 0):
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, Pmoy_y - Pmax_y)
                    else:
                        edgeD = AArete.Arete(Pmax_y, Pmoy_x, num, den, -1)
                    y = Pmoy_y
                    while (y < Pmax_y):
                        xG = edgeG.x
                        xD = edgeD.x
                        for x in range(xG, xD + 1):
                            t = -D / (A * x + B * y + C * d)
                            M3D = (t * x, t * y, t * d)
                            posx = zbuffer.dimx // 2 + x
                            posy = (zbuffer.dimy + 1) // 2 - 1 - y

                            if (M3D[2] > 0 and M3D[2] < zbuffer.acces(posx, posy)):
                                zbuffer.modif(posx, posy, M3D[2])
                                coul = self.calculercouleur(M3D, scene)
                                dessinerPoint((posx, posy), coul)
                                # dessinerPoint((posx,posy),self.facette.couleur)

                        edgeG.maj()
                        edgeD.maj()
                        y += 1
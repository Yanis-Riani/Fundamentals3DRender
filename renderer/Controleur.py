import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

import tkinter.filedialog
from typing import Any, Callable, List, Tuple

from . import Import_scene, Modele

# Type aliases for clarity (re-defined or imported from Modele for consistency)
Point2D = Tuple[int, int]
Point3D = Tuple[float, float, float]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]


class ControleurCourbes(object):
    """ Gere un ensemble de courbes. """
    def __init__(self) -> None:
        self.courbes: List[Modele.Courbe] = []
        self.scene: Import_scene.Donnees_scene | None = None  # sert pour l affichage des scenes (donnees importees)
        self.zbuffer: Modele.ZBuffer | None = None  # sert pour le z-buffer

    def ajouterCourbe(self, courbe: Modele.Courbe) -> None:
        """ Ajoute une courbe supplementaire.  """
        self.courbes.append(courbe)

    def dessiner(self, dessinerControle: DrawControlCallable, dessinerPoint: DrawPointCallable) -> None:
        """ Dessine les courbes. """
        # dessine les points de la courbe
        for courbe in self.courbes:
            courbe.dessinerPoints(dessinerPoint)
        # si la courbe peut etre remplie
        for courbe in self.courbes:
            if not isinstance(courbe, Modele.TriangleRempliZBuffer):
                courbe.remplir(dessinerPoint)
            else:
                if self.zbuffer is not None and self.scene is not None:
                    courbe.remplir(dessinerPoint, self.zbuffer, self.scene)

        # dessine les points de controle
        for courbe in self.courbes:
            if not isinstance(courbe, Modele.TriangleRempliZBuffer):
                courbe.dessinerControles(dessinerControle)

    def deplacerControle(self, ic: int, ip: int, point: Point2D) -> None:
        """ Deplace le point de controle a l'indice ip de la courbe a l'indice ic. """
        self.courbes[ic].controles[ip] = point

    def selectionnerControle(self, point: Point2D) -> Callable[[Point2D], None] | None:
        """ Trouve un point de controle proche d'un point donne. """
        xp, yp = point
        for ic in range(len(self.courbes)):
            for ip in range(len(self.courbes[ic].controles)):
                xc, yc = self.courbes[ic].controles[ip]
                if abs(xc - xp) < 4 and abs(yc - yp) < 4:
                    return lambda p: self.deplacerControle(ic, ip, p)
        return None

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

    def nouveauTriangleRempli(self) -> Callable[[Point2D], None]:
        """ Ajoute un nouveau triangle a remplir.
        Retourne une fonction permettant d'ajouter les points de controle. """
        trianglerempli = Modele.TriangleRempli((132, 165, 47))
        self.ajouterCourbe(trianglerempli)
        return trianglerempli.ajouterControle

    def nouvelleSceneFildefer(self, larg: int, haut: int) -> None:
        self.courbes = [] # Clear existing curves
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees
        d = self.scene.d  # distance de la camera a l ecran

        fic: str = "fic"
        indcptobj: int = -1
        while len(fic) > 0:  # tant que des fichiers objets selectionnes
            fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
            if len(fic) > 0:
                indcptobj += 1
                donnees.ajoute_objet(fic, indcptobj)

                self.scene = donnees

                # mettre objet  dans repere camera avec  translation differente de l objet Diamant et changement axes y<->z
                listesommetsdansreperecamera: List[Point3D] = []
                for som in self.scene.listeobjets[indcptobj].listesommets:
                    tx: float
                    ty: float
                    tz: float
                    if self.scene.listeobjets[indcptobj].nomobj == "Diamant":
                        tx = 150.0
                        ty = 150.0
                        tz = 2.2 * d
                    elif self.scene.listeobjets[indcptobj].nomobj == "Cube":
                        tx = 350.0
                        ty = 100.0
                        tz = 2 * d
                    else:
                        tx = 200.0
                        ty = 0.0
                        tz = 1.8 * d

                    yp = som[2] + ty
                    xp = -som[1] + tx
                    zp = som[0] + tz

                    listesommetsdansreperecamera.append((xp, yp, zp))

                listeprojete: List[Point2D] = []
                for som_3d in listesommetsdansreperecamera:
                    if som_3d[2] == 0: continue # Avoid division by zero
                    xp_2d = round(som_3d[0] * d / som_3d[2])
                    yp_2d = round(som_3d[1] * d / som_3d[2])
                    listeprojete.append((xp_2d, yp_2d))

                i = 0
                for tr in self.scene.listeobjets[indcptobj].listeindicestriangle:

                    droitemilieu = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu)
                    xp0 = larg // 2 + listeprojete[tr[0] - 1][0]
                    yp0 = (haut + 1) // 2 - 1 - listeprojete[tr[0] - 1][1]
                    point0 = (xp0, yp0)
                    droitemilieu.ajouterControle(point0)
                    xp1 = larg // 2 + listeprojete[tr[1] - 1][0]
                    yp1 = (haut + 1) // 2 - 1 - listeprojete[tr[1] - 1][1]
                    point1 = (xp1, yp1)
                    droitemilieu.ajouterControle(point1)

                    droitemilieu = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu)
                    droitemilieu.ajouterControle(point0)
                    xp2 = larg // 2 + listeprojete[tr[2] - 1][0]
                    yp2 = (haut + 1) // 2 - 1 - listeprojete[tr[2] - 1][1]
                    point2 = (xp2, yp2)
                    droitemilieu.ajouterControle(point2)

                    droitemilieu = Modele.DroiteMilieu()
                    self.ajouterCourbe(droitemilieu)
                    droitemilieu.ajouterControle(point1)
                    droitemilieu.ajouterControle(point2)

    def nouvelleScenePeintre(self, larg: int, haut: int) -> None:
        self.courbes = [] # Clear existing curves
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees
        d = self.scene.d  # distance de la camera a l ecran

        fic: str = "fic"
        indcptobj: int = -1
        while len(fic) > 0:  # tant que des fichiers objets selectionnes
            fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
            if len(fic) > 0:
                indcptobj += 1
                donnees.ajoute_objet(fic, indcptobj)

                self.scene = donnees

                # mettre objet  dans repere camera avec  translation differente de l objet Diamant et changement axes y<->z
                listesommetsdansreperecamera: List[Point3D] = []
                for som in self.scene.listeobjets[indcptobj].listesommets:
                    tx: float
                    ty: float
                    tz: float
                    if self.scene.listeobjets[indcptobj].nomobj == "Diamant":
                        tx = 150.0
                        ty = 150.0
                        tz = 2.2 * d
                    elif self.scene.listeobjets[indcptobj].nomobj == "Cube":
                        tx = 350.0
                        ty = 100.0
                        tz = 2 * d
                    else:
                        tx = 200.0
                        ty = 0.0
                        tz = 1.8 * d

                    yp = som[2] + ty
                    xp = -som[1] + tx
                    zp = som[0] + tz

                    listesommetsdansreperecamera.append((xp, yp, zp))

                listeprojete: List[Point2D] = []
                for som_3d in listesommetsdansreperecamera:
                    if som_3d[2] == 0: continue # Avoid division by zero
                    xp_2d = round(som_3d[0] * d / som_3d[2])
                    yp_2d = round(som_3d[1] * d / som_3d[2])
                    listeprojete.append((xp_2d, yp_2d))

                i = 0
                for tr in self.scene.listeobjets[indcptobj].listeindicestriangle:

                    trianglerempli = Modele.TriangleRempli(self.scene.listeobjets[indcptobj].listecouleurs[i])
                    i += 1
                    self.ajouterCourbe(trianglerempli)

                    xp0 = larg // 2 + listeprojete[tr[0] - 1][0]
                    yp0 = (haut + 1) // 2 - 1 - listeprojete[tr[0] - 1][1]
                    xp1 = larg // 2 + listeprojete[tr[1] - 1][0]
                    yp1 = (haut + 1) // 2 - 1 - listeprojete[tr[1] - 1][1]
                    xp2 = larg // 2 + listeprojete[tr[2] - 1][0]
                    yp2 = (haut + 1) // 2 - 1 - listeprojete[tr[2] - 1][1]

                    trianglerempli.ajouterControle((xp0, yp0))
                    trianglerempli.ajouterControle((xp1, yp1))
                    trianglerempli.ajouterControle((xp2, yp2))

    def initzbuffer(self) -> Callable[[int, int], None]:
        self.zbuffer = Modele.ZBuffer()
        # The returned function is a method of zbuffer, so it implicitly carries 'self.zbuffer'
        return self.zbuffer.alloc_init_zbuffer

    def nouvelleSceneZBuffer(self) -> None:
        self.courbes = [] # Clear existing curves
        donnees = Import_scene.Donnees_scene(os.path.join(ASSETS_DIR, "scenes", "Donnees_scene.sce"))
        self.scene = donnees

        d = self.scene.d  # distance de la camera a l ecran

        fic: str = "fic"
        indcptobj: int = -1
        while len(fic) > 0:
            fic = tkinter.filedialog.askopenfilename(title="Inserer l objet:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                                              filetypes=[("Fichiers Objets", "*.obj")])
            if len(fic) > 0:
                indcptobj += 1
                obj_texture = donnees.ajoute_objet(fic, indcptobj)

                self.scene = donnees

                # mettre objet  dans repere camera avec  translation differente de l objet Diamant et changement axes
                listesommetsdansreperecamera: List[Point3D] = []
                # listenormalesdansreperecamera=[]
                for som in self.scene.listeobjets[indcptobj].listesommets:
                    tx: float
                    ty: float
                    tz: float
                    if self.scene.listeobjets[indcptobj].nomobj == "Diamant":
                        tx = 150.0
                        ty = 150.0
                        tz = 2.2 * d

                    else:

                        if self.scene.listeobjets[indcptobj].nomobj == "Cube":
                            tx = 350.0
                            ty = 100.0
                            tz = 2 * d
                            # tz=1.5*d


                        else:
                            tx = 200.0
                            ty = 0.0
                            tz = 1.8 * d

                    yp = som[2] + ty
                    xp = -som[1] + tx
                    zp = som[0] + tz

                    listesommetsdansreperecamera.append((xp, yp, zp))


                listeprojete: List[Point2D] = []
                for som_3d in listesommetsdansreperecamera:
                    if som_3d[2] == 0: continue # Avoid division by zero
                    xp_2d = round(som_3d[0] * d / som_3d[2])
                    yp_2d = round(som_3d[1] * d / som_3d[2])

                    listeprojete.append((xp_2d, yp_2d))

                i = 0

                for tr in self.scene.listeobjets[indcptobj].listeindicestriangle:  # pour chaque triangle du polyedre
                    facette = Modele.Facette()

                    facette.indiceobjet = indcptobj
                    facette.texture_on = obj_texture

                    P1 = listesommetsdansreperecamera[tr[0] - 1]
                    P2 = listesommetsdansreperecamera[tr[1] - 1]
                    P3 = listesommetsdansreperecamera[tr[2] - 1]

                    facette.sommets.append(P1)
                    facette.sommets.append(P2)
                    facette.sommets.append(P3)


                    # equation du plan du triangle
                    A = (P2[1] - P1[1]) * (P3[2] - P1[2]) - ((P2[2] - P1[2]) * (P3[1] - P1[1]))
                    B = -((P2[0] - P1[0]) * (P3[2] - P1[2]) - ((P2[2] - P1[2]) * (P3[0] - P1[0])))
                    C = (P2[0] - P1[0]) * (P3[1] - P1[1]) - ((P2[1] - P1[1]) * (P3[0] - P1[0]))
                    D = -A * P1[0] - B * P1[1] - C * P1[2]

                    facette.normaleetplan = [A, B, C, D]
                    facette.couleur = self.scene.listeobjets[indcptobj].listecouleurs[i]
                    indicesdes3normalesauxsommets = self.scene.listeobjets[indcptobj].listeindicesnormales[i]
                    (n1x, n1y, n1z) = (self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[0] - 1][0],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[0] - 1][1],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[0] - 1][2])
                    (n2x, n2y, n2z) = (self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[1] - 1][0],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[1] - 1][1],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[1] - 1][2])
                    (n3x, n3y, n3z) = (self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[2] - 1][0],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[2] - 1][1],
                                      self.scene.listeobjets[indcptobj].listenormales[indicesdes3normalesauxsommets[2] - 1][2])

                    facette.normalesaux3sommets = [(n1x, n1y, n1z), (n2x, n2y, n2z), (n3x, n3y, n3z)]

                    if self.scene.listeobjets[indcptobj].texture_on:
                        indicesdes3texturesauxsommets = self.scene.listeobjets[indcptobj].listeindicestextures[i]
                        facette.coordtexturesaux3sommets = [
                            self.scene.listeobjets[indcptobj].listecoordtextures[indicesdes3texturesauxsommets[0] - 1],
                            self.scene.listeobjets[indcptobj].listecoordtextures[indicesdes3texturesauxsommets[1] - 1],
                            self.scene.listeobjets[indcptobj].listecoordtextures[indicesdes3texturesauxsommets[2] - 1]]

                    facette.coefs = self.scene.listeobjets[indcptobj].listecoefs[i]
                    i += 1


                    triangleremplizbuffer = Modele.TriangleRempliZBuffer()

                    self.ajouterCourbe(triangleremplizbuffer)
                    triangleremplizbuffer.ajouterfacette(facette)

                    """on ajoute les points dans le repere de la camera translate a z=d"""
                    point = listeprojete[tr[0] - 1]
                    triangleremplizbuffer.ajouterControle(point)
                    point = listeprojete[tr[1] - 1]
                    triangleremplizbuffer.ajouterControle(point)
                    point = listeprojete[tr[2] - 1]
                    triangleremplizbuffer.ajouterControle(point)

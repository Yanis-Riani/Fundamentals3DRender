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
    def __init__(self) -> None:
        self.courbes: List[Modele.Courbe] = []
        self.scene: Import_scene.Donnees_scene | None = None
        self.zbuffer: Modele.ZBuffer | None = None
        self.camera = camera.Camera(distance=400)
        self.loaded_objects: List[Tuple[Import_scene.Polyedre, bool]] = []
        self.current_rendering_mode: str = 'fildefer' # Default rendering mode

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

        # Iterate through all loaded objects
        for indcptobj_stored, (obj, obj_texture) in enumerate(self.loaded_objects):
            # Recalculate transformations for each object
            center = obj.get_center()
            center_translation = matrix.Matrix4.create_translation(vecteur3.Vecteur(-center.vx, -center.vy, -center.vz))
            transform_matrix = center_translation * view_matrix

            listesommetsdansreperecamera: List[Point3D] = []
            for som_coords in obj.listesommets:
                v = vecteur3.Vecteur(som_coords[0], som_coords[1], som_coords[2])
                v_transformed = transform_matrix.transform_point(v)
                listesommetsdansreperecamera.append((v_transformed.vx, v_transformed.vy, v_transformed.vz))

            listeprojete: List[Point2D] = []
            for som_3d in listesommetsdansreperecamera:
                if som_3d[2] == 0: continue
                xp_2d = round(som_3d[0] * d / som_3d[2])
                yp_2d = round(som_3d[1] * d / som_3d[2])
                listeprojete.append((xp_2d, yp_2d))

            # Apply rendering mode specific logic
            i = 0
            for tr in obj.listeindicestriangle:
                if mode == 'fildefer':
                    p0_idx, p1_idx, p2_idx = tr[0] - 1, tr[1] - 1, tr[2] - 1
                    point0 = (larg // 2 + listeprojete[p0_idx][0], (haut + 1) // 2 - 1 - listeprojete[p0_idx][1])
                    point1 = (larg // 2 + listeprojete[p1_idx][0], (haut + 1) // 2 - 1 - listeprojete[p1_idx][1])
                    point2 = (larg // 2 + listeprojete[p2_idx][0], (haut + 1) // 2 - 1 - listeprojete[p2_idx][1])

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
                    p0_idx, p1_idx, p2_idx = tr[0] - 1, tr[1] - 1, tr[2] - 1

                    trianglerempli = Modele.TriangleRempli(obj.listecouleurs[i]) # Using obj's colors directly
                    i += 1
                    self.ajouterCourbe(trianglerempli)

                    xp0 = larg // 2 + listeprojete[p0_idx][0]
                    yp0 = (haut + 1) // 2 - 1 - listeprojete[p0_idx][1]

                    xp1 = larg // 2 + listeprojete[p1_idx][0]
                    yp1 = (haut + 1) // 2 - 1 - listeprojete[p1_idx][1]

                    xp2 = larg // 2 + listeprojete[p2_idx][0]
                    yp2 = (haut + 1) // 2 - 1 - listeprojete[p2_idx][1]

                    trianglerempli.ajouterControle((xp0, yp0))
                    trianglerempli.ajouterControle((xp1, yp1))
                    trianglerempli.ajouterControle((xp2, yp2))

                elif mode == 'zbuffer':
                    # zbuffer init handled at the beginning of this method
                    if self.zbuffer is None: # Should not happen if zbuffer mode is active
                        continue

                    facette = Modele.Facette()
                    facette.indiceobjet = indcptobj_stored # Use the stored index
                    facette.texture_on = obj_texture # Use the stored texture flag

                    P1 = listesommetsdansreperecamera[tr[0] - 1]
                    P2 = listesommetsdansreperecamera[tr[1] - 1]
                    P3 = listesommetsdansreperecamera[tr[2] - 1]

                    facette.sommets.extend([P1, P2, P3])

                    A = (P2[1] - P1[1]) * (P3[2] - P1[2]) - ((P2[2] - P1[2]) * (P3[1] - P1[1]))
                    B = -((P2[0] - P1[0]) * (P3[2] - P1[2]) - ((P2[2] - P1[2]) * (P3[0] - P1[0])))
                    C = (P2[0] - P1[0]) * (P3[1] - P1[1]) - ((P2[1] - P1[1]) * (P3[0] - P1[0]))
                    D_plane = -A * P1[0] - B * P1[1] - C * P1[2]

                    facette.normaleetplan = [A, B, C, D_plane]
                    facette.couleur = obj.listecouleurs[i] # Using obj's colors

                    indicesdes3normalesauxsommets = obj.listeindicesnormales[i]
                    n1 = obj.listenormales[indicesdes3normalesauxsommets[0] - 1]
                    n2 = obj.listenormales[indicesdes3normalesauxsommets[1] - 1]
                    n3 = obj.listenormales[indicesdes3normalesauxsommets[2] - 1]
                    facette.normalesaux3sommets = [(n1[0], n1[1], n1[2]), (n2[0], n2[1], n2[2]), (n3[0], n3[1], n3[2])]

                    if obj.texture_on: # Using obj's texture_on flag
                        indicesdes3texturesauxsommets = obj.listeindicestextures[i]
                        t1 = obj.listecoordtextures[indicesdes3texturesauxsommets[0] - 1]
                        t2 = obj.listecoordtextures[indicesdes3texturesauxsommets[1] - 1]
                        t3 = obj.listecoordtextures[indicesdes3texturesauxsommets[2] - 1]
                        facette.coordtexturesaux3sommets = [t1, t2, t3]

                    facette.coefs = obj.listecoefs[i] # Using obj's coefs
                    i += 1

                    triangleremplizbuffer = Modele.TriangleRempliZBuffer()
                    self.ajouterCourbe(triangleremplizbuffer)
                    triangleremplizbuffer.ajouterfacette(facette)

                    p0_idx, p1_idx, p2_idx = tr[0] - 1, tr[1] - 1, tr[2] - 1
                    point0 = listeprojete[p0_idx]
                    point1 = listeprojete[p1_idx]
                    point2 = listeprojete[p2_idx]

                    triangleremplizbuffer.ajouterControle(point0)
                    triangleremplizbuffer.ajouterControle(point1)
                    triangleremplizbuffer.ajouterControle(point2)

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

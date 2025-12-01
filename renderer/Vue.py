from __future__ import annotations
import tkinter
from PIL import ImageTk, Image, ImageDraw
from tkinter.colorchooser import askcolor
from typing import List, Tuple, Callable, Any, Optional

from . import Controleur

# Type aliases for consistency and clarity
Point2D = Tuple[int, int]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]
# A callable that adds a control point, returning None.
AddControlCallable = Callable[[Point2D], None]


class VueCourbes(object):
    """ Gere l'affichage et la manipulation de courbe avec la bibliotheque Tkinter. """
    def __init__(self, largeur: int, hauteur: int) -> None:
        self.controleur: Controleur.ControleurCourbes = Controleur.ControleurCourbes()
        self.largeur: int = largeur
        self.hauteur: int = hauteur
        self.canvas: Optional[tkinter.Canvas] = None
        self.image: Optional[Image.Image] = None
        self.imageDraw: Optional[ImageDraw.ImageDraw] = None
        self.imageTk: Optional[ImageTk.PhotoImage] = None
        self.outilsCourant: Optional[AddControlCallable] = None # A single function or None
        self.outilsDeplacer: Optional[Callable[[Point2D], None]] = None

    def callbackButton1(self, event: tkinter.Event) -> None:
        """ Bouton gauche : utilise l'outils courant. """
        if not self.outilsCourant:
            # Type ignore because selectionnerControle can return None
            self.outilsDeplacer = self.controleur.selectionnerControle((event.x, event.y)) # type: ignore

    def callbackB1Motion(self, event: tkinter.Event) -> None:
        if self.outilsDeplacer:
            self.outilsDeplacer((event.x, event.y))
            self.majAffichage()

    def callbackButtonRelease1(self, event: tkinter.Event) -> None:
        """ Bouton gauche : utilise l'outils courant. """
        print(event.x, event.y)
        if self.outilsCourant:
            self.outilsCourant((event.x, event.y))
            self.majAffichage()
        if self.outilsDeplacer:
            self.outilsDeplacer = None

    def callbackButtonRelease3(self, event: tkinter.Event) -> None:
        """ Bouton droit : termine l'outils courant. """
        self.outilsCourant = None
        self.majAffichage()

    def callbackButton3(self, event: tkinter.Event) -> None:
        """ Bouton droit : termine l'outils courant. """
        self.outilsCourant = None
        self.majAffichage()

    def callbackNouveau(self) -> None:
        """ Supprime toutes les courbes. """
        self.controleur = Controleur.ControleurCourbes()
        self.majAffichage()

    def callbackHorizontale(self) -> None:
        """ Initialise l'outils courant pour ajouter une nouvelle horizontale. """
        self.outilsCourant = self.controleur.nouvelleHorizontale()

    def callbackVerticale(self) -> None:
        """ Initialise l'outils courant pour ajouter une nouvelle verticale. """
        self.outilsCourant = self.controleur.nouvelleVerticale()

    def callbackGD(self) -> None:
        """ Initialise l'outils courant pour ajouter un nouveau segment gauche et droite. """
        self.outilsCourant = self.controleur.nouvelleGD()

    def callbackMilieu(self) -> None:
        """ Initialise l'outils courant pour ajouter une nouvelle verticale. """
        self.outilsCourant = self.controleur.nouvellePointMilieu()

    def callbackTriangleRempli(self) -> None:
        """ Initialise l'outils courant pour ajouter un triangle rempli. """
        self.outilsCourant = self.controleur.nouveauTriangleRempli()

    def callbackNouvellesceneFildefer(self) -> None:
        """ Supprime toutes les courbes. """
        self.controleur = Controleur.ControleurCourbes()
        self.majAffichage()

        self.controleur.nouvelleSceneFildefer(self.largeur, self.hauteur)
        self.majAffichage()

    def callbackNouvellescenePeintre(self) -> None:
        """ Supprime toutes les courbes. """
        self.controleur = Controleur.ControleurCourbes()
        self.majAffichage()

        self.controleur.nouvelleScenePeintre(self.largeur, self.hauteur)
        self.majAffichage()

    def callbackNouvellesceneZBuffer(self) -> None:
        """ Supprime toutes les courbes. """
        self.controleur = Controleur.ControleurCourbes()
        self.majAffichage()

        init_zbuffer_func = self.controleur.initzbuffer()
        # passage des parametres de l image pour l allocation du zbuffer
        if init_zbuffer_func:
            init_zbuffer_func(self.largeur, self.hauteur)

        self.controleur.nouvelleSceneZBuffer()
        self.majAffichage()

    def majAffichage(self) -> None:
        """ Met a jour l'affichage.. """
        if self.imageDraw and self.image and self.canvas:
            # efface la zone de dession
            self.imageDraw.rectangle([0, 0, self.largeur, self.hauteur], fill='lightgrey')
            # dessine les courbes

            fonctionPoint: DrawPointCallable = lambda p, c: self.imageDraw.point(p, c)  # p le point, c la couleur
            fonctionControle: DrawControlCallable = lambda p: self.imageDraw.rectangle([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill='blue')
            self.controleur.dessiner(fonctionControle, fonctionPoint)
            # ImageTk : structure pour afficher l'image
            self.imageTk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(self.largeur / 2 + 1, self.hauteur / 2 + 1, image=self.imageTk)
        else:
            print("Warning: majAffichage called before image or canvas are initialized.")


    def executer(self) -> None:
        """ Initialise et lance le programme. """
        # fenetre principale
        fenetre = tkinter.Tk()
        fenetre.title("ASI1 : TP")
        fenetre.resizable(0, 0)
        # menu
        menu = tkinter.Menu(fenetre)
        fenetre.config(menu=menu)
        filemenu = tkinter.Menu(menu)
        menu.add_cascade(label="Fichier", menu=filemenu)
        filemenu.add_command(label="Nouveau", command=self.callbackNouveau)
        filemenu.add_separator()
        filemenu.add_command(label="Quitter", command=fenetre.destroy)
        toolsmenu = tkinter.Menu(menu)
        menu.add_cascade(label="Outils", menu=toolsmenu)
        toolsmenu.add_command(label="Ajouter une horizontale", command=self.callbackHorizontale)
        toolsmenu.add_command(label="Ajouter une verticale", command=self.callbackVerticale)
        toolsmenu.add_command(label="Ajouter un segment gauche-droite", command=self.callbackGD)
        toolsmenu.add_command(label="Ajouter un segment point milieu", command=self.callbackMilieu)
        toolsmenu.add_command(label="Ajouter un triangle rempli algo cours", command=self.callbackTriangleRempli)

        menu3D = tkinter.Menu(menu)
        menu.add_cascade(label="3D", menu=menu3D)
        menu3D.add_command(label="Import scene fil de fer", command=self.callbackNouvellesceneFildefer)
        menu3D.add_command(label="Import scene peintre", command=self.callbackNouvellescenePeintre)
        menu3D.add_command(label="Import scene ZBuffer", command=self.callbackNouvellesceneZBuffer)
        # Canvas : widget pour le dessin dans la fenetre principale
        self.canvas = tkinter.Canvas(fenetre, width=self.largeur, height=self.hauteur, bg='white')
        self.canvas.bind("<Button-1>", self.callbackButton1)
        self.canvas.bind("<B1-Motion>", self.callbackB1Motion)
        self.canvas.bind("<ButtonRelease-1>", self.callbackButtonRelease1)
        self.canvas.bind("<ButtonRelease-3>", self.callbackButtonRelease3)
        self.canvas.bind("<Button-3>", self.callbackButton3)
        self.canvas.pack()
        # Image : structure contenant les donnees de l'image manipule
        self.image = Image.new("RGB", (self.largeur, self.hauteur), 'lightgrey')
        # ImageDraw : structure pour manipuler l'image
        self.imageDraw = ImageDraw.Draw(self.image)
        # met a jour l'affichage
        self.majAffichage()
        # lance le programme
        fenetre.mainloop()

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
        self.middle_mouse_pressed: bool = False
        self.last_mouse_pos: Optional[Point2D] = None

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

    def callbackButton2(self, event: tkinter.Event) -> None:
        """ Middle mouse button press: start rotation. """
        self.middle_mouse_pressed = True
        self.last_mouse_pos = (event.x, event.y)

    def callbackButtonRelease2(self, event: tkinter.Event) -> None:
        """ Middle mouse button release: stop rotation. """
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None

    def callbackB2Motion(self, event: tkinter.Event) -> None:
        """ Middle mouse button drag: rotate camera or fine zoom. """
        if self.middle_mouse_pressed and self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            
            # Check for Ctrl key for fine zoom
            if event.state & 0x4: # Mask for Control key (Ctrl key is state 4)
                self.controleur.zoom_camera(dy) # Use dy for fine zoom
            else:
                self.controleur.rotate_camera(dx, dy)
            
            self.last_mouse_pos = (event.x, event.y)
            self.majAffichage()

    def callbackMouseWheel(self, event: tkinter.Event) -> None:
        """Mouse wheel scroll: coarse zoom."""
        # event.delta is typically 120 per click, or -120
        # Positive delta for zoom in, negative for zoom out
        self.controleur.zoom_camera(event.delta)
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

    def callback_importer(self) -> None:
        """Callback for the 'Importer Objet...' menu item."""
        # Reset controller for new import
        self.controleur = Controleur.ControleurCourbes()
        self.controleur.importer_objet(self.largeur, self.hauteur)
        self.majAffichage()

    def callback_set_mode(self, mode: str) -> None:
        """Callback to change the rendering mode of the currently loaded object(s)."""
        if self.controleur.loaded_objects:
            # Re-render with the new mode
            self.controleur.set_rendering_mode(self.largeur, self.hauteur, mode)
            self.majAffichage()
        else:
            print("No object loaded to change rendering mode.")


    def majAffichage(self) -> None:
        """ Met a jour l'affichage.. """
        if self.imageDraw and self.image and self.canvas:
            # efface la zone de dession
            self.imageDraw.rectangle([0, 0, self.largeur, self.hauteur], fill='lightgrey')
            # dessine les courbes

            # Rebuild courbes with current camera position and rendering mode
            self.controleur.rebuild_courbes(self.largeur, self.hauteur)

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
        
        menu3D.add_command(label="Importer Objet...", command=self.callback_importer)

        render_mode_menu = tkinter.Menu(menu3D)
        menu3D.add_cascade(label="Mode de Rendu", menu=render_mode_menu)
        render_mode_menu.add_command(label="Fil de Fer", command=lambda: self.callback_set_mode('fildefer'))
        render_mode_menu.add_command(label="Peintre", command=lambda: self.callback_set_mode('peintre'))
        render_mode_menu.add_command(label="Z-Buffer", command=lambda: self.callback_set_mode('zbuffer'))
        # Canvas : widget pour le dessin dans la fenetre principale
        self.canvas = tkinter.Canvas(fenetre, width=self.largeur, height=self.hauteur, bg='white')
        self.canvas.bind("<Button-1>", self.callbackButton1)
        self.canvas.bind("<B1-Motion>", self.callbackB1Motion)
        self.canvas.bind("<ButtonRelease-1>", self.callbackButtonRelease1)
        self.canvas.bind("<Button-3>", self.callbackButton3)
        self.canvas.bind("<ButtonRelease-3>", self.callbackButtonRelease3)
        self.canvas.bind("<Button-2>", self.callbackButton2)
        self.canvas.bind("<ButtonRelease-2>", self.callbackButtonRelease2)
        self.canvas.bind("<B2-Motion>", self.callbackB2Motion)
        self.canvas.bind("<MouseWheel>", self.callbackMouseWheel) # For scroll wheel zoom
        self.canvas.pack()
        # Image : structure contenant les donnees de l'image manipule
        self.image = Image.new("RGB", (self.largeur, self.hauteur), 'lightgrey')
        # ImageDraw : structure pour manipuler l'image
        self.imageDraw = ImageDraw.Draw(self.image)
        # met a jour l'affichage
        self.majAffichage()
        # lance le programme
        fenetre.mainloop()

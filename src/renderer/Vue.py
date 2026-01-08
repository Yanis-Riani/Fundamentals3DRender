from __future__ import annotations
import tkinter
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tkinter.colorchooser import askcolor
from typing import List, Tuple, Callable, Any, Optional

from . import Controleur

# Type aliases for consistency and clarity
Point2D = Tuple[int, int]
Couleur = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Couleur], None]
DrawControlCallable = Callable[[Point2D], None]
AddControlCallable = Callable[[Point2D], None]


class VueCourbes(object):
    """ Gere l'affichage et la manipulation de courbe avec la bibliotheque Tkinter. """
    def __init__(self, largeur: int, hauteur: int) -> None:
        self.controleur: Controleur.ControleurCourbes = Controleur.ControleurCourbes(self)
        self.largeur: int = largeur
        self.hauteur: int = hauteur
        self.canvas: Optional[tkinter.Canvas] = None
        self.image: Optional[Image.Image] = None
        self.imageDraw: Optional[ImageDraw.ImageDraw] = None
        self.imageTk: Optional[ImageTk.PhotoImage] = None
        self.outilsCourant: Optional[AddControlCallable] = None 
        self.middle_mouse_pressed: bool = False
        self.last_mouse_pos: Optional[Point2D] = None

    def callbackButton1(self, event: tkinter.Event) -> None:
        """ Bouton gauche : utilise l'outils courant ou selectionne. """
        
        if not self.outilsCourant:
            # Selection logic
            self.controleur.selectionnerControle((event.x, event.y), self.controleur.mode)
            self.majAffichage()

    def callbackButtonRelease1(self, event: tkinter.Event) -> None:
        if self.outilsCourant:
            self.outilsCourant((event.x, event.y))
            self.majAffichage()

    def callbackButtonRelease3(self, event: tkinter.Event) -> None:
        self.outilsCourant = None
        self.majAffichage()

    def callbackButton3(self, event: tkinter.Event) -> None:
        """ Bouton droit : termine l'outils courant ou annule grab. """
        if self.controleur.grab_state["active"]:
            self.controleur.cancel_grab()
            return
            
        self.outilsCourant = None
        self.majAffichage()

    def callbackButton2(self, event: tkinter.Event) -> None:
        self.middle_mouse_pressed = True
        self.last_mouse_pos = (event.x, event.y)

    def callbackButtonRelease2(self, event: tkinter.Event) -> None:
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None

    def callbackB2Motion(self, event: tkinter.Event) -> None:
        if self.middle_mouse_pressed and self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            if event.state & 0x4: 
                self.controleur.zoom_camera(dy) 
            else:
                self.controleur.rotate_camera(dx, dy)
            self.last_mouse_pos = (event.x, event.y)
            self.majAffichage()

    def callbackMouseWheel(self, event: tkinter.Event) -> None:
        self.controleur.zoom_camera(event.delta)
        self.majAffichage()

    def callbackToggleMode(self, event: tkinter.Event) -> None:
        if self.controleur.mode == 'viewer':
            self.controleur.mode = 'edit'
            print("Switched to Edit Mode")
        else:
            self.controleur.mode = 'viewer'
            print("Switched to Viewer Mode")
        self.majAffichage()
        
    def callbackKeyPress(self, event: tkinter.Event) -> None:
        """Handle keyboard shortcuts."""
        key = event.keysym.lower()
        
        if key == 'g':
            if self.controleur.grab_state["active"]:
                self.controleur.confirm_grab()
            else:
                self.controleur.start_grab_mode()
        elif key in ['x', 'y', 'z']:
            shift_pressed = (event.state & 0x1) != 0 # Check Shift mask
            self.controleur.toggle_axis_constraint(key, shift_pressed)
        elif key == 'return':
            self.controleur.confirm_grab()
        elif key == 'escape':
            self.controleur.cancel_grab()
            
    def callbackMotion(self, event: tkinter.Event) -> None:
        """Handle mouse movement for grab mode."""
        if self.controleur.grab_state["active"]:
            self.controleur.update_grab(event.x, event.y)


    def callbackNouveau(self) -> None:
        self.controleur = Controleur.ControleurCourbes(self)
        self.majAffichage()

    def callbackHorizontale(self) -> None:
        self.outilsCourant = self.controleur.nouvelleHorizontale()

    def callbackVerticale(self) -> None:
        self.outilsCourant = self.controleur.nouvelleVerticale()

    def callbackGD(self) -> None:
        self.outilsCourant = self.controleur.nouvelleGD()

    def callbackMilieu(self) -> None:
        self.outilsCourant = self.controleur.nouvellePointMilieu()

    def callback_importer(self) -> None:
        self.controleur = Controleur.ControleurCourbes(self)
        self.controleur.importer_objet(self.largeur, self.hauteur)
        self.majAffichage()

    def callback_set_mode(self, mode: str) -> None:
        if self.controleur.loaded_objects:
            self.controleur.set_rendering_mode(self.largeur, self.hauteur, mode)
            self.majAffichage()
        else:
            print("No object loaded to change rendering mode.")


    def majAffichage(self) -> None:
        if self.imageDraw and self.image and self.canvas:
            self.imageDraw.rectangle([0, 0, self.largeur, self.hauteur], fill='lightgrey')
            
            self.controleur.rebuild_courbes(self.largeur, self.hauteur)

            fonctionPoint: DrawPointCallable = lambda p, c: self.imageDraw.point(p, c) 
            fonctionControle: DrawControlCallable = lambda p: self.imageDraw.rectangle([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill='blue')
            self.controleur.dessiner(fonctionControle, fonctionPoint)

            # Highlight selected vertex if in 'edit' mode
            if self.controleur.mode == 'edit' and self.controleur.selected_vertex_index is not None:
                obj_idx, v_idx = self.controleur.selected_vertex_index
                if 0 <= v_idx < len(self.controleur.projected_vertices_2d):
                    selected_2d_pos = self.controleur.projected_vertices_2d[v_idx]
                    selection_size = 5 
                    self.imageDraw.rectangle([selected_2d_pos[0] - selection_size,
                                               selected_2d_pos[1] - selection_size,
                                               selected_2d_pos[0] + selection_size,
                                               selected_2d_pos[1] + selection_size],
                                              fill='red', outline='red')
            
            self.imageTk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(self.largeur / 2 + 1, self.hauteur / 2 + 1, image=self.imageTk)

            # --- UI Overlay ---
            self.canvas.delete("ui_overlay") # Clear previous overlay

            if self.controleur.mode == 'edit':
                # Blue Border
                self.canvas.create_rectangle(2, 2, self.largeur, self.hauteur, outline='blue', width=4, tags="ui_overlay")

            # Status Text
            mode_text = f"MODE: {self.controleur.mode.upper()} (TAB)"
            
            # Add constraint info if grabbing
            if self.controleur.grab_state["active"]:
                constraint = self.controleur.grab_state["constraint"]
                if constraint:
                    # Map constraint code to user-friendly text
                    constraint_text = ""
                    if constraint == 'x': constraint_text = "X Lock"
                    elif constraint == 'y': constraint_text = "Y Lock"
                    elif constraint == 'z': constraint_text = "Z Lock"
                    elif constraint == 'shift_x': constraint_text = "Y/Z Lock"
                    elif constraint == 'shift_y': constraint_text = "X/Z Lock"
                    elif constraint == 'shift_z': constraint_text = "X/Y Lock"
                    
                    if constraint_text:
                        mode_text += f" ● {constraint_text}"

            # Draw text shadow for better visibility
            self.canvas.create_text(11, 11, text=mode_text, anchor="nw", fill="black", font=("Arial", 12, "bold"), tags="ui_overlay")
            self.canvas.create_text(10, 10, text=mode_text, anchor="nw", fill="white", font=("Arial", 12, "bold"), tags="ui_overlay")

        else:
            print("Warning: majAffichage called before image or canvas are initialized.")


    def executer(self) -> None:
        fenetre = tkinter.Tk()
        fenetre.title("ASI1 : TP")
        fenetre.resizable(0, 0)
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


        menu3D = tkinter.Menu(menu)
        menu.add_cascade(label="3D", menu=menu3D)
        
        menu3D.add_command(label="Importer Objet...", command=self.callback_importer)

        render_mode_menu = tkinter.Menu(menu3D)
        menu3D.add_cascade(label="Mode de Rendu", menu=render_mode_menu)
        render_mode_menu.add_command(label="Fil de Fer", command=lambda: self.callback_set_mode('fildefer'))
        render_mode_menu.add_command(label="Peintre", command=lambda: self.callback_set_mode('peintre'))
        render_mode_menu.add_command(label="Z-Buffer", command=lambda: self.callback_set_mode('zbuffer'))
        
        self.canvas = tkinter.Canvas(fenetre, width=self.largeur, height=self.hauteur, bg='white')
        self.canvas.bind("<Button-1>", self.callbackButton1)
        self.canvas.bind("<ButtonRelease-1>", self.callbackButtonRelease1)
        self.canvas.bind("<Button-3>", self.callbackButton3)
        self.canvas.bind("<ButtonRelease-3>", self.callbackButtonRelease3)
        self.canvas.bind("<Button-2>", self.callbackButton2)
        self.canvas.bind("<ButtonRelease-2>", self.callbackButtonRelease2)
        self.canvas.bind("<B2-Motion>", self.callbackB2Motion)
        self.canvas.bind("<MouseWheel>", self.callbackMouseWheel) 
        self.canvas.bind("<Key-Tab>", self.callbackToggleMode)
        
        # New Bindings for Blender-style interaction
        # Bind KeyPress to the *window* (fenetre) to capture keys globally when focused
        fenetre.bind("<Key>", self.callbackKeyPress)
        self.canvas.bind("<Motion>", self.callbackMotion)
        self.canvas.pack()
        self.canvas.focus_set() # Ensure canvas has focus for key events if needed

        self.image = Image.new("RGB", (self.largeur, self.hauteur), 'lightgrey')
        self.imageDraw = ImageDraw.Draw(self.image)
        self.majAffichage()
        fenetre.mainloop()

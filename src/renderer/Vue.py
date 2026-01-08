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
        
        # Selection Box State
        self.selection_start: Optional[Point2D] = None
        self.selection_current: Optional[Point2D] = None

        # Notification State
        self.notification_text: Optional[str] = None
        self.notification_timer_id: Optional[str] = None

    def show_notification(self, text: str, duration: int = 2000) -> None:
        """Shows a temporary notification text."""
        self.notification_text = text
        self.majAffichage()
        
        # Cancel previous timer if exists
        if self.notification_timer_id:
            self.canvas.after_cancel(self.notification_timer_id)
            self.notification_timer_id = None
            
        # Schedule clear
        self.notification_timer_id = self.canvas.after(duration, self._clear_notification)

    def _clear_notification(self) -> None:
        self.notification_text = None
        self.notification_timer_id = None
        self.majAffichage()

    def callbackButton1(self, event: tkinter.Event) -> None:
        """ Bouton gauche : valide transform ou commence selection. """
        
        # Priority: Transform Confirmation
        if self.controleur.transform_state["active"]:
            self.controleur.confirm_transform()
            return

        if not self.outilsCourant:
            # Start Box Selection logic
            self.selection_start = (event.x, event.y)
            self.selection_current = (event.x, event.y)
            
            shift_pressed = (event.state & 0x1) != 0
            self.controleur.selectionnerControle((event.x, event.y), self.controleur.mode, shift_pressed)
            self.majAffichage()

    def callbackButtonRelease1(self, event: tkinter.Event) -> None:
        if self.outilsCourant:
            self.outilsCourant((event.x, event.y))
            self.majAffichage()
        else:
            # Finish Box Selection
            if self.selection_start and self.selection_current:
                dx = abs(self.selection_start[0] - self.selection_current[0])
                dy = abs(self.selection_start[1] - self.selection_current[1])
                if dx > 2 or dy > 2:
                    shift_pressed = (event.state & 0x1) != 0
                    rect = (self.selection_start[0], self.selection_start[1], 
                            self.selection_current[0], self.selection_current[1])
                    self.controleur.selectionner_zone(rect, self.controleur.mode, shift_pressed)
            
            self.selection_start = None
            self.selection_current = None
            self.majAffichage()

    def callbackButtonRelease3(self, event: tkinter.Event) -> None:
        self.outilsCourant = None
        self.majAffichage()

    def callbackButton3(self, event: tkinter.Event) -> None:
        """ Bouton droit : termine l'outils courant ou annule transform. """
        if self.controleur.transform_state["active"]:
            self.controleur.cancel_transform()
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
            
            # State masks (Tkinter specific)
            # Shift = 0x1 (1)
            # Ctrl = 0x4 (4)
            # Alt = 0x20000 (131072) or similar? Depends on OS. usually we check bit 0, 2 etc.
            
            is_shift = (event.state & 0x1) != 0
            is_ctrl = (event.state & 0x4) != 0
            
            if is_shift:
                self.controleur.pan_camera(dx, dy)
            elif is_ctrl:
                self.controleur.zoom_camera(dy)
            else:
                self.controleur.rotate_camera(dx, dy)
                
            self.last_mouse_pos = (event.x, event.y)
            self.majAffichage()

    def callbackMouseWheel(self, event: tkinter.Event) -> None:
        self.controleur.zoom_camera(event.delta)
        self.majAffichage()

    def callbackToggleMode(self, event: tkinter.Event) -> None:
        if self.controleur.transform_state["active"]:
            self.controleur.cancel_transform()

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
        state = event.state
        ctrl_pressed = (state & 0x4) != 0
        
        # Undo/Redo
        if ctrl_pressed and key == 'z':
            if not self.controleur.transform_state["active"] and self.controleur.get_undo_count() > 0:
                self.controleur.perform_undo()
                self.show_notification("Undo")
            return
        if ctrl_pressed and key == 'y':
            if not self.controleur.transform_state["active"] and self.controleur.get_redo_count() > 0:
                self.controleur.perform_redo()
                self.show_notification("Redo")
            return

        # Tools
        if key == 'g':
            if self.controleur.transform_state["active"]:
                if self.controleur.transform_state["mode"] == 'grab':
                    self.controleur.cancel_transform()
                else:
                    self.controleur.cancel_transform()
                    self.controleur.start_transform_mode('grab', (self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()))
            else:
                self.controleur.start_transform_mode('grab', (self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()))
        
        elif key == 'r':
            if self.controleur.transform_state["active"]:
                if self.controleur.transform_state["mode"] == 'rotate':
                    self.controleur.cancel_transform()
                else:
                    self.controleur.cancel_transform()
                    self.controleur.start_transform_mode('rotate', (self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()))
            else:
                self.controleur.start_transform_mode('rotate', (self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()))

        elif key in ['x', 'y', 'z']:
            shift_pressed = (state & 0x1) != 0 
            self.controleur.toggle_axis_constraint(key, shift_pressed)
        
        elif key == 'return':
            self.controleur.confirm_transform()
        elif key == 'escape':
            self.controleur.cancel_transform()
            
    def callbackMotion(self, event: tkinter.Event) -> None:
        """Handle mouse movement for transform mode AND box selection."""
        if self.controleur.transform_state["active"]:
            self.controleur.update_transform(event.x, event.y)
            return

        if self.selection_start:
            self.selection_current = (event.x, event.y)
            self.majAffichage()


    def callbackNouveau(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur = Controleur.ControleurCourbes(self)
        self.majAffichage()

    def callbackHorizontale(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.outilsCourant = self.controleur.nouvelleHorizontale()

    def callbackVerticale(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.outilsCourant = self.controleur.nouvelleVerticale()

    def callbackGD(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.outilsCourant = self.controleur.nouvelleGD()

    def callbackMilieu(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.outilsCourant = self.controleur.nouvellePointMilieu()

    def callback_importer(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur = Controleur.ControleurCourbes(self)
        self.controleur.importer_objet(self.largeur, self.hauteur)
        self.majAffichage()

    def callback_set_mode(self, mode: str) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
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

            # Highlight selected vertices
            if self.controleur.mode == 'edit':
                for obj_idx, v_idx in self.controleur.selected_vertices:
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
            self.canvas.delete("ui_overlay") 

            # Selection Box
            if self.selection_start and self.selection_current:
                self.canvas.create_rectangle(self.selection_start[0], self.selection_start[1],
                                             self.selection_current[0], self.selection_current[1],
                                             outline='white', dash=(4, 4), width=1, tags="ui_overlay")

            # Status
            theme_color = 'blue' if self.controleur.mode == 'edit' else 'black'
            
            # --- Bottom Left Status (History) ---
            undo_count = self.controleur.get_undo_count()
            
            history_text = f"{undo_count} edit (ctrl+z/y)"
            if self.notification_text:
                history_text += f" ● {self.notification_text.lower()}"
            
            # Draw History Text (Bottom Left)
            self.canvas.create_text(10, self.hauteur - 15, text=history_text, anchor="sw", fill=theme_color, font=("Arial", 10, "bold"), tags="ui_overlay")


            # --- Top Left Status (Mode + Tool) ---
            status_parts = [f"{self.controleur.mode} (tab)"]
            
            if self.controleur.transform_state["active"]:
                mode_name = self.controleur.transform_state["mode"] # grab or rotate
                status_parts.append(mode_name)
                
                constraint = self.controleur.transform_state["constraint"]
                if constraint:
                    c_text = ""
                    if constraint == 'x': c_text = "x"
                    elif constraint == 'y': c_text = "y"
                    elif constraint == 'z': c_text = "z"
                    elif constraint == 'shift_x': c_text = "y/z"
                    elif constraint == 'shift_y': c_text = "x/z"
                    elif constraint == 'shift_z': c_text = "x/y"
                    if c_text:
                        status_parts.append(f"{c_text} lock")
            
            text_str = " ● ".join(status_parts).lower()

            # Text & Border Logic
            text_id = self.canvas.create_text(35, 2, text=text_str, anchor="nw", fill=theme_color, font=("Arial", 10, "bold"), tags="ui_overlay")
            bbox = self.canvas.bbox(text_id) 
            
            if bbox:
                margin = 5
                x1, y1 = margin, 10
                x2, y2 = self.largeur - margin, self.hauteur - margin
                radius = 8 
                gap_pad = 5
                
                gap_start = max(x1 + radius, bbox[0] - gap_pad)
                gap_end = min(x2 - radius, bbox[2] + gap_pad)
                
                self.canvas.create_arc(x1, y1, x1+2*radius, y1+2*radius, start=90, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                
                if gap_start > x1 + radius:
                    self.canvas.create_line(x1+radius, y1, gap_start, y1, fill=theme_color, width=2, tags="ui_overlay")
                if gap_end < x2 - radius:
                    self.canvas.create_line(gap_end, y1, x2-radius, y1, fill=theme_color, width=2, tags="ui_overlay")
                
                self.canvas.create_arc(x2-2*radius, y1, x2, y1+2*radius, start=0, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x2, y1+radius, x2, y2-radius, fill=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_arc(x2-2*radius, y2-2*radius, x2, y2, start=270, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x2-radius, y2, x1+radius, y2, fill=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_arc(x1, y2-2*radius, x1+2*radius, y2, start=180, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x1, y2-radius, x1, y1+radius, fill=theme_color, width=2, tags="ui_overlay")

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
        # ... tools menu ...
        toolsmenu = tkinter.Menu(menu)
        menu.add_cascade(label="Outils", menu=toolsmenu)
        toolsmenu.add_command(label="Ajouter une horizontale", command=self.callbackHorizontale)
        # ... other tools ...

        menu3D = tkinter.Menu(menu)
        menu.add_cascade(label="3D", menu=menu3D)
        
        menu3D.add_command(label="Importer Objet...", command=self.callback_importer)

        render_mode_menu = tkinter.Menu(menu3D)
        menu3D.add_cascade(label="Mode de Rendu", menu=render_mode_menu)
        render_mode_menu.add_command(label="Fil de Fer", command=lambda: self.callback_set_mode('fildefer'))
        render_mode_menu.add_command(label="Peintre", command=lambda: self.callback_set_mode('peintre'))
        render_mode_menu.add_command(label="Z-Buffer", command=lambda: self.callback_set_mode('zbuffer'))
        
        self.canvas = tkinter.Canvas(fenetre, width=self.largeur, height=self.hauteur, bg='white', highlightthickness=0, borderwidth=0)
        self.canvas.bind("<Button-1>", self.callbackButton1)
        self.canvas.bind("<ButtonRelease-1>", self.callbackButtonRelease1)
        self.canvas.bind("<Button-3>", self.callbackButton3)
        self.canvas.bind("<ButtonRelease-3>", self.callbackButtonRelease3)
        self.canvas.bind("<Button-2>", self.callbackButton2)
        self.canvas.bind("<ButtonRelease-2>", self.callbackButtonRelease2)
        self.canvas.bind("<B2-Motion>", self.callbackB2Motion)
        self.canvas.bind("<MouseWheel>", self.callbackMouseWheel) 
        self.canvas.bind("<Key-Tab>", self.callbackToggleMode)
        
        fenetre.bind("<Key>", self.callbackKeyPress)
        self.canvas.bind("<Motion>", self.callbackMotion)
        self.canvas.pack()
        self.canvas.focus_set() 

        self.image = Image.new("RGB", (self.largeur, self.hauteur), 'lightgrey')
        self.imageDraw = ImageDraw.Draw(self.image)
        self.majAffichage()
        fenetre.mainloop()
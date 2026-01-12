from __future__ import annotations
import tkinter
import os
import functools
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tkinter.colorchooser import askcolor
from typing import List, Tuple, Callable, Any, Optional

from . import Controleur

# Global Asset Path Helper
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")

# Type aliases for consistency and clarity
Point2D = Tuple[int, int]
Color = Tuple[int, int, int]
DrawPointCallable = Callable[[Point2D, Color], None]
DrawControlCallable = Callable[[Point2D], None]
AddControlCallable = Callable[[Point2D], None]


class CurveView(object):
    """Manages display and interaction with curves using Tkinter."""
    def __init__(self, width: int, height: int) -> None:
        self.controleur: Controleur.CurveController = Controleur.CurveController(self)
        self.width: int = width
        self.height: int = height
        self.canvas: Optional[tkinter.Canvas] = None
        self.image: Optional[Image.Image] = None
        self.imageDraw: Optional[ImageDraw.ImageDraw] = None
        self.imageTk: Optional[ImageTk.PhotoImage] = None
        self.current_tool: Optional[AddControlCallable] = None 
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
        self.update_display()
        if self.notification_timer_id:
            self.canvas.after_cancel(self.notification_timer_id)
        self.notification_timer_id = self.canvas.after(duration, self._clear_notification)

    def _clear_notification(self) -> None:
        self.notification_text = None
        self.notification_timer_id = None
        self.update_display()

    def callbackButton1(self, event: tkinter.Event) -> None:
        self.canvas.focus_set()
        if self.controleur.transform_state["active"]:
            self.controleur.confirm_transform()
            return
        if not self.current_tool:
            self.selection_start = (event.x, event.y)
            self.selection_current = (event.x, event.y)
            shift_pressed = (event.state & 0x1) != 0
            self.controleur.select_control((event.x, event.y), self.controleur.mode, shift_pressed)
            self.update_display()

    def callbackButtonRelease1(self, event: tkinter.Event) -> None:
        if self.current_tool:
            self.current_tool((event.x, event.y))
            self.update_display()
        else:
            if self.selection_start and self.selection_current:
                dx = abs(self.selection_start[0] - self.selection_current[0])
                dy = abs(self.selection_start[1] - self.selection_current[1])
                if dx > 2 or dy > 2:
                    shift_pressed = (event.state & 0x1) != 0
                    rect = (self.selection_start[0], self.selection_start[1], 
                            self.selection_current[0], self.selection_current[1])
                    self.controleur.select_area(rect, self.controleur.mode, shift_pressed)
            self.selection_start = None
            self.selection_current = None
            self.update_display()

    def callbackButtonRelease3(self, event: tkinter.Event) -> None:
        self.current_tool = None
        self.update_display()

    def callbackButton3(self, event: tkinter.Event) -> None:
        if self.controleur.transform_state["active"]:
            self.controleur.cancel_transform()
            return
        self.current_tool = None
        self.update_display()

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
            is_shift = (event.state & 0x1) != 0
            is_ctrl = (event.state & 0x4) != 0
            if is_shift: self.controleur.pan_camera(dx, dy)
            elif is_ctrl: self.controleur.zoom_camera(dy)
            else: self.controleur.rotate_camera(dx, dy)
            self.last_mouse_pos = (event.x, event.y)
            self.update_display()

    def callbackMouseWheel(self, event: tkinter.Event) -> None:
        self.controleur.zoom_camera(event.delta)
        self.update_display()

    def callbackToggleMode(self, event: tkinter.Event) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur.mode = 'edit' if self.controleur.mode == 'viewer' else 'viewer'
        self.update_display()
        
    def callbackKeyPress(self, event: tkinter.Event) -> None:
        key = event.keysym.lower(); raw_key = event.keysym; state = event.state; ctrl_pressed = (state & 0x4) != 0
        if ctrl_pressed and key == 'z':
            if not self.controleur.transform_state["active"] and self.controleur.get_undo_count() > 0:
                self.controleur.perform_undo(); self.show_notification("Undo")
            return
        if ctrl_pressed and key == 'y':
            if not self.controleur.transform_state["active"] and self.controleur.get_redo_count() > 0:
                self.controleur.perform_redo(); self.show_notification("Redo")
            return
        if key == 'g':
            if self.controleur.transform_state["active"] and self.controleur.transform_state["mode"] == 'grab':
                self.controleur.cancel_transform()
            else:
                if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
                self.controleur.start_transform_mode('grab', (event.x, event.y))
        elif key == 'r':
            if self.controleur.transform_state["active"] and self.controleur.transform_state["mode"] == 'rotate':
                self.controleur.cancel_transform()
            else:
                if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
                self.controleur.start_transform_mode('rotate', (event.x, event.y))
        elif key in ['x', 'y', 'z']:
            shift_pressed = (state & 0x1) != 0 
            self.controleur.toggle_axis_constraint(key, shift_pressed)
        elif key == 'return': 
            if self.controleur.transform_state["active"]: self.controleur.confirm_transform()
            else: self.controleur.ui_mode_apply(self.width, self.height); self.update_display()
        elif key == 'escape': self.controleur.cancel_transform()
        elif key == 'up' or raw_key == 'Up':
            self.controleur.ui_mode_cycle(-1); self.update_display()
        elif key == 'down' or raw_key == 'Down':
            self.controleur.ui_mode_cycle(1); self.update_display()
            
    def callbackMotion(self, event: tkinter.Event) -> None:
        if self.controleur.transform_state["active"]: self.controleur.update_transform(event.x, event.y); return
        if self.selection_start: self.selection_current = (event.x, event.y); self.update_display()

    def on_new(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur = Controleur.CurveController(self); self.update_display()

    def on_horizontal(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.current_tool = self.controleur.new_horizontal()

    def on_vertical(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.current_tool = self.controleur.new_vertical()

    def on_left_right(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.current_tool = self.controleur.new_left_right()

    def on_midpoint(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.current_tool = self.controleur.new_midpoint_line()

    def on_import(self) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur.import_object(self.width, self.height); self.update_display()

    def on_load_example(self, file_path: str) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        self.controleur.import_object_direct(self.width, self.height, file_path); self.update_display()

    def on_set_mode(self, mode: str) -> None:
        if self.controleur.transform_state["active"]: self.controleur.cancel_transform()
        if self.controleur.loaded_objects: self.controleur.set_rendering_mode(self.width, self.height, mode); self.update_display()

    def update_display(self) -> None:
        if self.imageDraw and self.image and self.canvas:
            self.imageDraw.rectangle([0, 0, self.width, self.height], fill='lightgrey')
            self.controleur.rebuild_curves(self.width, self.height)
            
            # 1. Draw Grid Lines
            for x1, y1, x2, y2, color in self.controleur.grid_lines_2d:
                self.imageDraw.line([(x1, y1), (x2, y2)], fill=color)

            # 2. Draw Solid Faces
            for pts, color in self.controleur.solid_faces_2d:
                self.imageDraw.polygon(pts, fill=color, outline=None)

            # 3. Draw Wireframe Lines
            for x1, y1, x2, y2, color in self.controleur.wireframe_lines_2d:
                self.imageDraw.line([(x1, y1), (x2, y2)], fill=color)

            # 4. Draw Curves
            draw_point: DrawPointCallable = lambda p, c: self.imageDraw.point(p, c) 
            draw_control: DrawControlCallable = lambda p: self.imageDraw.rectangle([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill='blue')
            self.controleur.draw(draw_control, draw_point)

            # 5. Draw Edit Mode Vertices
            if self.controleur.mode == 'edit':
                for obj_idx, v_idx in enumerate(range(len(self.controleur.projected_vertices_2d))):
                    if self.controleur.current_rendering_mode in ['peintre', 'zbuffer'] and v_idx not in self.controleur.visible_vertices:
                        continue
                    pos = self.controleur.projected_vertices_2d[v_idx]
                    is_selected = (0, v_idx) in self.controleur.selected_vertices
                    color = (0, 255, 255) if is_selected else (0, 0, 0)
                    r = 3 
                    self.imageDraw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline=color)
            
            self.imageTk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(self.width / 2, self.height / 2, image=self.imageTk)
            self.canvas.delete("ui_overlay") 
            if self.selection_start and self.selection_current:
                self.canvas.create_rectangle(self.selection_start[0], self.selection_start[1], self.selection_current[0], self.selection_current[1], outline='white', dash=(4, 4), width=1, tags="ui_overlay")
            
            theme_color = 'blue' if self.controleur.mode == 'edit' else 'black'
            undo_count = self.controleur.get_undo_count()
            history_text = f"{undo_count} edit (ctrl+z/y)"
            if self.notification_text: history_text += f" ● {self.notification_text.lower()}"
            self.canvas.create_text(10, self.height - 15, text=history_text, anchor="sw", fill=theme_color, font=("Arial", 10, "bold"), tags="ui_overlay")
            
            status_parts = [f"{self.controleur.mode} (tab)"]
            if self.controleur.transform_state["active"]:
                mode_name = self.controleur.transform_state["mode"]
                status_parts.append(mode_name)
                constraint = self.controleur.transform_state["constraint"]
                if constraint:
                    c_text = {"x":"x", "y":"y", "z":"z", "shift_x":"y/z", "shift_y":"x/z", "shift_z":"x/y"}.get(constraint, "")
                    if c_text: status_parts.append(f"{c_text} lock")
            text_str = " ● ".join(status_parts).lower()
            text_id = self.canvas.create_text(35, 2, text=text_str, anchor="nw", fill=theme_color, font=("Arial", 10, "bold"), tags="ui_overlay")
            
            bbox = self.canvas.bbox(text_id) 
            if bbox:
                margin = 5; x1, y1 = margin, 10; x2, y2 = self.width - margin, self.height - margin; radius = 8; gap_pad = 5
                gap_start = max(x1 + radius, bbox[0] - gap_pad); gap_end = min(x2 - radius, bbox[2] + gap_pad)
                self.canvas.create_arc(x1, y1, x1+2*radius, y1+2*radius, start=90, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                if gap_start > x1 + radius: self.canvas.create_line(x1+radius, y1, gap_start, y1, fill=theme_color, width=2, tags="ui_overlay")
                if gap_end < x2 - radius: self.canvas.create_line(gap_end, y1, x2-radius, y1, fill=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_arc(x2-2*radius, y1, x2, y1+2*radius, start=0, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x2, y1+radius, x2, y2-radius, fill=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_arc(x2-2*radius, y2-2*radius, x2, y2, start=270, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x2-radius, y2, x1+radius, y2, fill=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_arc(x1, y2-2*radius, x1+2*radius, y2, start=180, extent=90, style="arc", outline=theme_color, width=2, tags="ui_overlay")
                self.canvas.create_line(x1, y2-radius, x1, y1+radius, fill=theme_color, width=2, tags="ui_overlay")
            
            # Draw Render Mode Menu (Top Right)
            menu_x, menu_y = self.width - 10, 10
            for idx, (label, key) in enumerate(self.controleur.available_modes):
                is_selected = (idx == self.controleur.ui_mode_index)
                is_active = (key == self.controleur.current_rendering_mode)
                prefix = "> " if is_selected else "  "
                suffix = " [ON]" if is_active else ""
                display_text = f"{prefix}{label}{suffix}"
                font_weight = "bold" if is_selected else "normal"
                fill_color = "red" if is_selected else theme_color
                self.canvas.create_text(menu_x, menu_y + (idx * 15), text=display_text, anchor="ne", fill=fill_color, font=("Consolas", 10, font_weight), tags="ui_overlay")

        else: print("Warning: update_display called before image or canvas are initialized.")

    def run(self) -> None:
        fenetre = tkinter.Tk(); fenetre.title("ASI1 : TP"); fenetre.resizable(0, 0)
        
        menu = tkinter.Menu(fenetre)
        fenetre.config(menu=menu)
        
        menu.add_command(label="New", command=self.on_new)
        menu.add_command(label="Import (.obj)", command=self.on_import)

        examples_menu = tkinter.Menu(menu, tearoff=0)
        menu.add_cascade(label="Examples", menu=examples_menu)
        
        scenes_dir = os.path.join(ASSETS_DIR, "scenes")
        if os.path.exists(scenes_dir):
            for filename in os.listdir(scenes_dir):
                if filename.lower().endswith(".obj"):
                    file_path = os.path.join(scenes_dir, filename)
                    examples_menu.add_command(
                        label=filename, 
                        command=functools.partial(self.on_load_example, file_path)
                    )
        
        self.canvas = tkinter.Canvas(fenetre, width=self.width, height=self.height, bg='white', highlightthickness=0, borderwidth=0)
        self.canvas.bind("<Button-1>", self.callbackButton1)
        self.canvas.bind("<ButtonRelease-1>", self.callbackButtonRelease1)
        self.canvas.bind("<Button-3>", self.callbackButton3)
        self.canvas.bind("<ButtonRelease-3>", self.callbackButtonRelease3)
        self.canvas.bind("<Button-2>", self.callbackButton2)
        self.canvas.bind("<ButtonRelease-2>", self.callbackButtonRelease2)
        self.canvas.bind("<B2-Motion>", self.callbackB2Motion)
        self.canvas.bind("<MouseWheel>", self.callbackMouseWheel)
        self.canvas.bind("<Key-Tab>", self.callbackToggleMode)
        self.canvas.bind("<Key>", self.callbackKeyPress)
        self.canvas.bind("<Motion>", self.callbackMotion)
        
        self.canvas.pack()
        self.canvas.focus_set()
        
        self.image = Image.new("RGB", (self.width, self.height), 'lightgrey')
        self.imageDraw = ImageDraw.Draw(self.image)
        self.update_display()
        fenetre.mainloop()
from __future__ import annotations
from typing import List, Tuple, Any
import os
import tkinter.filedialog
from PIL import Image
from . import vector3

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")


class Polyhedron:
    def get_center(self) -> vector3.Vector3:
        """Calculates the geometric center of the object's vertices."""
        if not self.vertices:
            return vector3.Vector3(0, 0, 0)
        
        sum_x = sum(v[0] for v in self.vertices)
        sum_y = sum(v[1] for v in self.vertices)
        sum_z = sum(v[2] for v in self.vertices)
        
        num_vertices = len(self.vertices)
        return vector3.Vector3(sum_x / num_vertices, sum_y / num_vertices, sum_z / num_vertices)

    def save_to_obj(self, filename: str) -> None:
        """Exports the polyhedron to a Wavefront .obj file."""
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# Exported by Fundamentals3DRender\n")
            if self.texture_on:
                f.write(f"# texture_enabled: true\n")
            f.write(f"o {self.name if self.name else 'Object'}\n")
            
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write texture coordinates
            for vt in self.texture_coords:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
                
            # Write normals
            for vn in self.normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
                
            # Write faces
            for i, tri in enumerate(self.triangle_indices):
                f.write("f")
                for j, v_idx in enumerate(tri):
                    # OBJ indices are 1-based. Our internal indices seem to be 1-based (from parsing code).
                    # 'f v/vt/vn' or 'f v//vn' or 'f v/vt' or 'f v'
                    
                    # Check if we have texture indices for this face
                    vt_idx = ""
                    if self.texture_indices and i < len(self.texture_indices) and len(self.texture_indices[i]) > j:
                        vt_val = self.texture_indices[i][j]
                        if vt_val != 0: # 0 often means no index in some parsers, but here checking >0 safer
                             vt_idx = str(vt_val)
                    
                    # Check if we have normal indices for this face
                    vn_idx = ""
                    if self.normal_indices and i < len(self.normal_indices) and len(self.normal_indices[i]) > j:
                         vn_idx = str(self.normal_indices[i][j])
                    
                    if vt_idx or vn_idx:
                        f.write(f" {v_idx}/{vt_idx}/{vn_idx}")
                    else:
                        f.write(f" {v_idx}")
                f.write("\n")

    def __init__(self) -> None:
        self.name: str = ""
        self.vertices: List[List[float]] = []
        self.normals: List[List[float]] = []
        self.texture_coords: List[List[float]] = []
        self.triangle_indices: List[List[int]] = []    # for each triangle
        self.normal_indices: List[List[int]] = []      # for each triangle
        self.texture_indices: List[List[int]] = []     # for each triangle
        self.colors: List[Tuple[int, int, int]] = []
        self.coeffs: List[Tuple[float, float, float, int]] = []
        self.texture_on: bool = False
        self.texture_image: List[Any] = []
        self.texture_size: List[Tuple[int, int]] = []
        self.object_index: int = -1


class SceneData:

    def __init__(self, filename: str = "") -> None:
        self.ambient_intensity: float = 0.0
        self.camera_distance: float = 0.0
        self.lights: List[List[float]] = []    # list of quadruplets (posx,posy,posz,Intensity)
        self.objects: List[Polyhedron] = []  # list of polyhedra

        if not filename:
            return

        with open(filename, "r") as f:
            for line in f:
                if line.startswith('a'):
                    data = line.rstrip('\n\r').split(" ")
                    self.ambient_intensity = float(data[1])

                if line.startswith('d'):
                    data = line.rstrip('\n\r').split(" ")
                    self.camera_distance = float(data[1])

                if line.startswith('l '):
                    lum = []
                    data = line.rstrip('\n\r').split(" ")
                    lum.append(float(data[1]))
                    lum.append(float(data[2]))
                    lum.append(float(data[3]))
                    lum.append(float(data[4]))
                    self.lights.append(lum)

    def add_object(self, filename: str = "", object_idx: int = -1) -> bool:
        try:
            with open(filename, "r") as f:
                color: Tuple[int, int, int] = (0, 0, 0)
                coeffs: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
                tex_coords_exist = False

                poly = Polyhedron()
                poly.object_index = object_idx

                # Auto-detect format:
                # Legacy format: First line is 0 or 1 (int), second line is name.
                # Standard format: Starts with comments #, 'v', 'o', etc.
                
                start_pos = f.tell()
                first_line = f.readline()
                is_legacy = False
                
                try:
                    val = int(first_line.strip())
                    if val in [0, 1]:
                        is_legacy = True
                except ValueError:
                    pass
                
                f.seek(start_pos) # Rewind to start

                if is_legacy:
                    line1 = f.readline()
                    line2 = f.readline()
                    poly.name = line2.strip()

                    if int(line1) == 1:  # if polyhedron is texturable
                        texture_file = tkinter.filedialog.askopenfilename(
                            title="Associate a texture with the object?:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                            filetypes=[("Textures", "*.jpg; *.png; *.bmp")]
                        )
                        if len(texture_file) > 0:
                            poly.texture_on = True
                            img = Image.open(texture_file)
                            print(f"Texture dimensions {img.size}")
                            mat = list(img.getdata())
                            poly.texture_image = mat
                            poly.texture_size.append(img.size)
                else:
                    # Standard OBJ: Set default name from filename
                    poly.name = os.path.basename(filename).split('.')[0]
                    
                    # Check for texture hint in first few lines
                    texture_hint_found = False
                    # Read header lines (peek)
                    header_lines = []
                    curr_pos = f.tell()
                    for _ in range(10): # Check first 10 lines max
                        l = f.readline()
                        if not l: break
                        header_lines.append(l)
                        if "texture_enabled: true" in l:
                            texture_hint_found = True
                            break
                    f.seek(curr_pos) # Rewind to start of file (after legacy check)

                    if texture_hint_found:
                         texture_file = tkinter.filedialog.askopenfilename(
                            title="Associate a texture with the object?:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                            filetypes=[("Textures", "*.jpg; *.png; *.bmp")]
                        )
                         if len(texture_file) > 0:
                            poly.texture_on = True
                            img = Image.open(texture_file)
                            print(f"Texture dimensions {img.size}")
                            mat = list(img.getdata())
                            poly.texture_image = mat
                            poly.texture_size.append(img.size)


                for line in f:
                    if line.startswith('v '):
                        vertex = []
                        data = line.rstrip('\n\r').split(" ")
                        vertex.append(float(data[1]))
                        vertex.append(float(data[2]))
                        vertex.append(float(data[3]))
                        poly.vertices.append(vertex)

                    elif line.startswith('vn'):
                        norm = []
                        data = line.rstrip('\n\r').split(" ")
                        norm.append(float(data[1]))
                        norm.append(float(data[2]))
                        norm.append(float(data[3]))
                        poly.normals.append(norm)

                    elif line.startswith('vt'):
                        txt = []
                        tex_coords_exist = True
                        data = line.rstrip('\n\r').split(" ")
                        txt.append(float(data[1]))
                        txt.append(float(data[2]))
                        poly.texture_coords.append(txt)

                    elif line.startswith('c'):
                        data = line.rstrip('\n\r').split(" ")
                        for s in data:
                            if s != "c" and s != "":
                                color_data = s.split("/")
                                red = int(color_data[0])
                                green = int(color_data[1])
                                blue = int(color_data[2])
                                color = (red, green, blue)
                    
                    elif line.startswith('k'):
                        data = line.rstrip('\n\r').split(" ")
                        for s in data:
                            if s != "k" and s != "":
                                coeff_data = s.split("/")
                                ka = float(coeff_data[0])
                                krd = float(coeff_data[1])
                                krs = float(coeff_data[2])
                                ns = int(coeff_data[3])
                                coeffs = (ka, krd, krs, ns)
                    
                    elif line.startswith('o '):
                        # Update name if explicit 'o' tag is found
                        poly.name = line[2:].strip()

                    elif line.startswith('f'):
                        tri_indices = []
                        norm_indices = []
                        tex_indices = []
                        data = line.rstrip('\n\r').split(" ")
                        for s in data:
                            if s != "f" and s != "":
                                face_data = s.split("/")
                                tri_indices.append(int(face_data[0]))
                                if tex_coords_exist and len(face_data) > 1 and face_data[1]:
                                    tex_indices.append(int(face_data[1]))
                                else:
                                    if not texture_hint_found:
                                        poly.texture_on = False
                                    else:
                                        # Keep texture on, use 0 (or 1) as placeholder
                                        tex_indices.append(1) # Default to 1 to avoid crash if list not empty

                        poly.triangle_indices.append(tri_indices)
                        poly.texture_indices.append(tex_indices)
                        poly.normal_indices.append(norm_indices)
                        poly.colors.append(color)
                        poly.coeffs.append(coeffs)

                self.objects.append(poly)
                return poly.texture_on
        except (IOError, IndexError, ValueError) as e:
            print(f"Error processing file {filename}: {e}")
            return False

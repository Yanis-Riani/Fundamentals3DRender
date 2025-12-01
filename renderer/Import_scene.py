from __future__ import annotations
from typing import List, Tuple, Any
import os
import tkinter.filedialog
from PIL import Image

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")


class Polyedre():
    def __init__(self) -> None:
        self.nomobj: str = ""
        self.listesommets: List[List[float]] = []
        self.listenormales: List[List[float]] = []
        self.listecoordtextures: List[List[float]] = []
        self.listeindicestriangle: List[List[int]] = []    # pour chaque triangle
        self.listeindicesnormales: List[List[int]] = []    # pour chaque triangle
        self.listeindicestextures: List[List[int]] = []    # pour chaque triangle
        self.listecouleurs: List[Tuple[int, int, int]] = []
        self.listecoefs: List[Tuple[float, float, float, int]] = []
        self.texture_on: bool = False
        self.texture_ima: List[Any] = []
        self.texture_size: List[Tuple[int, int]] = []
        self.indice_objet: int = -1


class Donnees_scene():

    def __init__(self, nomfic: str = "") -> None:
        self.Ia: float = 0.0
        self.d: float = 0.0
        self.listelum: List[List[float]] = []    # liste de quadruplets (posx,posy,posz,IS)
        self.listeobjets: List[Polyedre] = []  # liste des polyedres

        if not nomfic:
            return

        with open(nomfic, "r") as fichier:
            for ligne in fichier:
                if ligne.startswith('a'):
                    donnees = ligne.rstrip('\n\r').split(" ")
                    self.Ia = float(donnees[1])

                if ligne.startswith('d'):
                    donnees = ligne.rstrip('\n\r').split(" ")
                    self.d = float(donnees[1])

                if ligne.startswith('l '):
                    lum = []
                    donnees = ligne.rstrip('\n\r').split(" ")
                    lum.append(float(donnees[1]))
                    lum.append(float(donnees[2]))
                    lum.append(float(donnees[3]))
                    lum.append(float(donnees[4]))
                    self.listelum.append(lum)

    def ajoute_objet(self, nomfic: str = "", indcptobj: int = -1) -> bool:
        from PIL import Image


        try:
            with open(nomfic, "r") as fichier:
                coul: Tuple[int, int, int] = (0, 0, 0)
                coefs: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
                coordonnees_texture_existent = False

                poly = Polyedre()
                poly.indice_objet = indcptobj

                ligne1 = fichier.readline()
                ligne2 = fichier.readline()
                poly.nomobj = ligne2.strip()

                if int(ligne1) == 1:  # si le polyedre est texturable
                    fichiertexture = filedialog.askopenfilename(
                        title="Associer une texture a l objet?:", initialdir=os.path.join(ASSETS_DIR, "scenes"),
                        filetypes=[("Textures", "*.jpg; *.png; *.bmp")]
                    )
                    if len(fichiertexture) > 0:
                        poly.texture_on = True
                        img = Image.open(fichiertexture)
                        print(f"Dimensions de la texture {img.size}")
                        mat = list(img.getdata())
                        poly.texture_ima = mat
                        poly.texture_size.append(img.size)

                for ligne in fichier:
                    if ligne.startswith('v '):
                        sommet = []
                        donnees = ligne.rstrip('\n\r').split(" ")
                        sommet.append(float(donnees[1]))
                        sommet.append(float(donnees[2]))
                        sommet.append(float(donnees[3]))
                        poly.listesommets.append(sommet)

                    elif ligne.startswith('vn'):
                        norm = []
                        donnees = ligne.rstrip('\n\r').split(" ")
                        norm.append(float(donnees[1]))
                        norm.append(float(donnees[2]))
                        norm.append(float(donnees[3]))
                        poly.listenormales.append(norm)

                    elif ligne.startswith('vt'):
                        txt = []
                        coordonnees_texture_existent = True
                        donnees = ligne.rstrip('\n\r').split(" ")
                        txt.append(float(donnees[1]))
                        txt.append(float(donnees[2]))
                        poly.listecoordtextures.append(txt)

                    elif ligne.startswith('c'):
                        don = ligne.rstrip('\n\r').split(" ")
                        for chaine in don:
                            if chaine != "c" and chaine != "":
                                donnees_coul = chaine.split("/")
                                rouge = int(donnees_coul[0])
                                vert = int(donnees_coul[1])
                                bleu = int(donnees_coul[2])
                                coul = (rouge, vert, bleu)
                    
                    elif ligne.startswith('k'):
                        don = ligne.rstrip('\n\r').split(" ")
                        for chaine in don:
                            if chaine != "k" and chaine != "":
                                donnees_coefs = chaine.split("/")
                                ka = float(donnees_coefs[0])
                                krd = float(donnees_coefs[1])
                                krs = float(donnees_coefs[2])
                                ns = int(donnees_coefs[3])
                                coefs = (ka, krd, krs, ns)

                    elif ligne.startswith('f'):
                        indicestriangle = []
                        indicesnormalesautriangle = []
                        indicescoordtextureautriangle = []
                        don = ligne.rstrip('\n\r').split(" ")
                        for chaine in don:
                            if chaine != "f" and chaine != "":
                                donnees_face = chaine.split("/")
                                indicestriangle.append(int(donnees_face[0]))
                                if coordonnees_texture_existent and len(donnees_face) > 1 and donnees_face[1]:
                                    indicescoordtextureautriangle.append(int(donnees_face[1]))
                                else:
                                    poly.texture_on = False
                                if len(donnees_face) > 2:
                                    indicesnormalesautriangle.append(int(donnees_face[2]))

                        poly.listeindicestriangle.append(indicestriangle)
                        poly.listeindicestextures.append(indicescoordtextureautriangle)
                        poly.listeindicesnormales.append(indicesnormalesautriangle)
                        poly.listecouleurs.append(coul)
                        poly.listecoefs.append(coefs)

                self.listeobjets.append(poly)
                return poly.texture_on
        except (IOError, IndexError, ValueError) as e:
            print(f"Error processing file {nomfic}: {e}")
            return False

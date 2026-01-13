# Fundamentals3DRender

Une application de rendu 3D interactive développée en Python. Initialement conçu comme un projet académique, cet outil a été enrichi pour offrir une interface complète permettant d'importer, de manipuler et d'exporter des modèles au format Wavefront `.obj`. Le moteur supporte plusieurs techniques de rendu classiques (Fil de fer, Algorithme du peintre, Z-Buffer) avec une gestion du shading (Phong/Gouraud) et du texturage.

## Installation

1.  Cloner le dépôt :
    ```sh
    git clone https://github.com/Yanis-Riani/Fundamentals3DRender.git
    ```
2.  Installer les dépendances :
    ```sh
    pip install -r requirements.txt
    ```
3.  Lancer l'application :
    ```sh
    python main.py
    ```

## Utilisation

### Gestion des Modèles

*   **Import** : Menu **"Import (.obj)"** pour charger un modèle. Le menu **"Examples"** permet de charger rapidement des scènes prédéfinies.
*   **Export** : Menu **"Export (.obj)"** pour sauvegarder vos modifications dans un nouveau fichier `.obj`.

### Contrôles Caméra

*   **Rotation** : Maintenir le **bouton central** de la souris et glisser.
*   **Pan (Déplacement)** : **Shift** + **bouton central**.
*   **Zoom** : **Molette** ou **Ctrl** + **bouton central**.

### Édition d'Objets

*   **Mode Édition** : Appuyez sur `Tab` pour basculer entre le mode **Viewer** et **Edit**.
*   **Sélection** :
    *   Clic gauche pour sélectionner un sommet.
    *   Maintenir `Shift` pour une sélection multiple ou pour tracer un rectangle de sélection.
*   **Transformations** :
    *   **Déplacement (Grab)** : Appuyez sur `G`.
    *   **Rotation** : Appuyez sur `R`.
*   **Contraintes d'Axes** :
    *   `X`, `Y` ou `Z` : Bloque la transformation sur cet axe unique.
    *   **Shift + X**, **Shift + Y** ou **Shift + Z** : Bloque l'axe choisi pour transformer sur les deux autres (ex: `Shift + X` pour contraindre sur les axes **Y** et **Z**).
*   **Valider/Annuler** :
    *   `Entrée` ou Clic gauche pour valider.
    *   `Échap` ou Clic droit pour annuler.
*   **Historique** :
    *   **Undo** : `Ctrl + Z`
    *   **Redo** : `Ctrl + Y`

### Modes de Rendu

*   Utilisez les flèches **Haut/Bas** pour naviguer dans la liste des modes (Wireframe, Solid, Z-Buffer) en haut à droite.
*   Appuyez sur `Entrée` pour appliquer le mode sélectionné.

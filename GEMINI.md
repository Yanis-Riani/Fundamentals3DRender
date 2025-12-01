# GEMINI.md

## Project Overview

This project is a 3D rendering application developed in Python. It provides a graphical user interface for loading 3D models from Wavefront `.obj` files and rendering them using several classic techniques. The application is built using the standard `tkinter` library for its UI and `Pillow` for image manipulation.

The architecture follows a Model-View-Controller (MVC) pattern:
*   **Model (`Modele.py`)**: Defines the data structures for 2D and 3D shapes. It contains the core rendering logic, including implementations of scanline rasterization for triangles, Z-buffering, Phong shading (Gouraud style), and texture mapping.
*   **View (`Vue.py`)**: Manages the `tkinter` window, canvas, and menu systems. It captures user input (mouse clicks) and is responsible for drawing the final image to the screen.
*   **Controller (`Controleur.py`)**: Acts as the intermediary. It handles UI events from the View, uses the `Import_scene` module to load 3D data, performs 3D transformations (translation, perspective projection), and populates the Model with rendering primitives (lines or triangles).

**Key Features:**
*   Parses `.obj` files for vertices, normals, texture coordinates, and faces.
*   Parses custom `.sce` files for scene-level data like lighting and camera settings.
*   Renders scenes in three modes:
    1.  **Wireframe (`fil de fer`)**: Draws the edges of the 3D models.
    2.  **Painter's Algorithm (`peintre`)**: A depth-sorting based algorithm (though the implementation seems to rely on the order of triangles).
    3.  **Z-Buffer**: Accurate per-pixel depth testing for correct visibility.
*   Implements the Phong lighting model with ambient, diffuse, and specular components.
*   Supports texture mapping on models.

## Building and Running

### Dependencies

The project relies on the **Pillow** library. There is no `requirements.txt` file, so it must be installed manually.

```bash
pip install Pillow
```

### Running the Application

The application can be started by running the `main.py` script.

```bash
python src/main.py
```

Upon running, a `tkinter` window will appear. Use the "3D" menu to load a scene. The application will open a file dialog to select one or more `.obj` files from the `src/scenes/` directory.

### Project Structure

*   `src/main.py`: The main entry point of the application.
*   `src/Vue.py`: Handles the GUI (Tkinter).
*   `src/Controleur.py`: The controller in the MVC structure.
*   `src/Modele.py`: The model in the MVC structure, contains rendering algorithms.
*   `src/Import_scene.py`: Handles loading of `.obj` and `.sce` files.
*   `src/vecteur3.py`: A utility class for 3D vector operations.
*   `src/AArete.py`: A helper class for the scanline rendering algorithm.
*   `src/scenes/`: Contains `.obj` models, textures, and scene definition files.

## Development Conventions

*   **Language:** Python 2.x (inferred from `print` statements without parentheses).
*   **Styling:** The code uses a mix of French and English for variable names and comments (e.g., `courbes`, `dessiner`, `zbuffer`). It generally follows PEP8 for naming (`snake_case` for functions and variables, `PascalCase` for classes), but with some inconsistencies.
*   **Dependencies:** No dependency management file (like `requirements.txt`) is present.
*   **Testing:** There are no apparent unit tests or testing framework configured.
*   **Hardcoded Paths:** There is a hardcoded absolute path in `src/Controleur.py` within the `nouvelleSceneZBuffer` function. This should be refactored to use relative paths.
    ```python
    donnees=Import_scene.Donnees_scene("C:/Users/Yanis/Downloads/3d python/solution/TP7_Texture_Corrigé/TP7_Texture_Corrigé/scenes/Donnees_scene.sce")
    ```

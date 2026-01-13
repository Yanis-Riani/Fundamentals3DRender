# Fundamentals3DRender

This project is a 3D rendering application originally developed as a school project and subsequently enhanced for better usability and features. It provides a graphical interface to import, view, edit, and export 3D models in the Wavefront OBJ format, supporting various rendering techniques like Wireframe, Painter's Algorithm, and Z-Buffer.

## Install

1.  Clone the repository
    ```sh
    git clone https://github.com/Yanis-Riani/Fundamentals3DRender.git
    ```
2.  Install dependencies
    ```sh
    pip install -r requirements.txt
    ```
3.  Run the application
    ```sh
    python main.py
    ```

## Usage

### Managing Models

*   **Import**: Select **"Import (.obj)"** from the menu to load a 3D model. You can also use **"Examples"** to load pre-defined scenes.
*   **Export**: Select **"Export (.obj)"** to save the currently loaded and modified object to a new OBJ file.

### Camera Controls

*   **Rotate**: Drag with the **Middle Mouse Button**.
*   **Pan**: Hold **Shift** + drag with the **Middle Mouse Button**.
*   **Zoom**: Use the **Mouse Wheel** or hold **Ctrl** + drag with the **Middle Mouse Button**.

### Editing Objects

*   **Toggle Mode**: Press `Tab` to switch between **Viewer** and **Edit** modes.
*   **Selection**:
    *   Click to select a vertex.
    *   Hold `Shift` to multi-select or drag to box-select.
*   **Transformations**:
    *   **Move (Grab)**: Press `G`.
    *   **Rotate**: Press `R`.
*   **Constraints**:
    *   Press `X`, `Y`, or `Z` to constrain to a single axis.
    *   Press `Shift + X`, `Shift + Y`, or `Shift + Z` to constrain to a plane (e.g., `Shift + Z` locks the Z axis, allowing movement on the XY plane).
*   **Confirm/Cancel**:
    *   Press `Enter` or Click to confirm.
    *   Press `Esc` or Right Click to cancel.
*   **History**:
    *   **Undo**: `Ctrl + Z`
    *   **Redo**: `Ctrl + Y`

### Rendering Modes

*   **Cycle Modes**: Use **Up/Down Arrow** keys to select a rendering mode (Wireframe, Solid, Z-Buffer).
*   **Apply**: Press `Enter` to activate the selected mode.
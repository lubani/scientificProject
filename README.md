# 2-Phase Stefan Problem Solver using PINNs

## Overview

This project involves a standalone Python 3 desktop application designed to solve the 2-phase Stefan problem using Physics-Informed Neural Networks (PINNs) along with analytical and numerical methods. The application integrates TensorFlow, Keras, CUDA for GPU acceleration, and Tkinter for the user interface.

## Author

Luba Ira - [lubani@ac.sce.ac.il](mailto:lubani@ac.sce.ac.il)

## Project Description

The application simulates the latent heat thermal energy storage in lunar regolith, which is crucial for future lunar settlement. The PINN approach allows for efficient and accurate predictions of phase change materials (PCMs) behavior under varying thermal conditions.

## Features

- **Physics-Informed Neural Networks (PINNs):** Dual-branch network architecture for temperature and boundary predictions.
- **Numerical Methods:** Explicit and implicit numerical solutions for handling the Stefan problem.
- **Analytical Solutions:** Provides baseline solutions for comparison.
- **Graphical User Interface (GUI):** Built with Tkinter for easy interaction.
- **GPU Acceleration:** Utilizes CUDA for computational efficiency.

## Files

- **`interface3.py`:** The main file containing the GUI and the application logic.
- **`deepLearning.py`:** Contains the PINN model architecture and training functions.
- **`backend.py`:** Includes utility functions and additional computational methods.

### How to Run the Program

1. **Ensure Prerequisites Are Met:**
   - **Docker Desktop**: Make sure Docker Desktop is installed and running on your system.
   - **Xming (Windows users)**: Install and run Xming to display GUI applications. Configure it to allow external connections.
   - **X Server for Linux/macOS**: Ensure X11 (Linux) or XQuartz (macOS) is installed and running for GUI display.

2. **Pull the Docker image from Google Cloud Container Registry:**
    
    ```commandline
   docker pull gcr.io/sonic-cumulus-357519/scientific-project
    ```
3. **Run the Docker container:**

    ```commandline
   docker run -it --rm -e DISPLAY=host.docker.internal:0.0 gcr.io/sonic-cumulus-357519/scientific-project
    ```
   
4. **Display Configuration:**
   - Ensure the DISPLAY environment variable is correctly set for your operating system to connect to the X server (e.g., DISPLAY=host.docker.internal:0.0 for Windows users).

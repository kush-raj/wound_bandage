# 🩹 Real-Time Wound Bandage Overlay System
A real-time computer vision application built using Python, OpenCV, and NumPy that detects red-colored wound areas from a webcam feed and overlays a transparent bandage image dynamically. This project simulates a simple Augmented Reality (AR) medical assistance system.

# 🚀 Features
1. 🎥 Real-time webcam processing
2. 🔴 Accurate red color detection using HSV color space
3. 🧹 Noise reduction with morphological operations
4. 📍 Contour detection for wound localization
5. 🩹 Transparent PNG bandage overlay with alpha blending
6. 📉 Smooth tracking to reduce jitter
7. 🖥 Side-by-side original and processed output

# 🛠 Tech Stack
1. Python
2. OpenCV
3. NumPy

# ⚙️ How It Works
1. Captures live video from webcam.
2. Converts frames to HSV color space for better red detection.
3. Creates a mask for red regions using dual HSV ranges.
4. Cleans noise using morphological operations.
5. Detects the largest contour as the wound.
6. Applies smoothing to reduce shaking.
7. Dynamically resizes and overlays a transparent bandage image.

# ▶️ Installation & Run
pip install opencv-python numpy
env\Scripts\activate
python main.py

# 📸 Output Preview

<img width="284" height="262" alt="wound" src="https://github.com/user-attachments/assets/f3228e12-0c0d-4ef3-afce-4fec9b0884f9" />

# 📌 Note
For best results, ensure good lighting, working webcam and a clearly visible red wound area.

#  Air Whiteboard Pro

Draw in the air using your hands — no mouse or touchscreen needed!  
This project uses **Mediapipe** for real-time hand tracking and **OpenCV** for drawing on a virtual canvas.

---

##  Demo
Draw by pointing your **index finger** at the screen.  
Change colors or brush sizes by moving your finger over the on-screen buttons.  
Clear the screen by hovering your finger over the red “CLEAR” button.  

---

##  Features
-  Real-time **hand detection** with Mediapipe  
-  Draw on an **invisible whiteboard** using your finger  
-  Choose from **five colors**  
-  Adjust **brush thickness**  
-  Instantly **clear the canvas**  
-  **Mirrored camera** view for natural interaction  

---

##  Requirements
Make sure you have Python installed (3.8+ recommended), then install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Air-Whiteboard-Pro.git
   cd Air-Whiteboard-Pro
   ```

2. Run the Python script:
   ```bash
   python air_whiteboard_pro.py
   ```

3. Use your **index finger** to draw in the air!
   - Move your finger over color circles → change color  
   - Move over gray buttons → change brush size  
   - Hover over the red rectangle → clear screen  
   - Press **Q** to quit  

---

##  Controls Summary

| Action | Description |
|--------|-------------|
| 🟢 Color Buttons | Change the drawing color |
| ⚪ Size Buttons | Change the brush size |
| 🔴 CLEAR Button | Erase all drawings |
| 👉 One Finger Up | Draw mode |
| ✋ Multiple Fingers Up | Stop drawing |
| ⌨️ Press Q | Quit program |

---

##  Tech Stack
- [OpenCV](https://opencv.org/) – Image processing and display  
- [Mediapipe](https://developers.google.com/mediapipe) – Hand tracking  
- [NumPy](https://numpy.org/) – Canvas and matrix operations  

---

##  License
This project is open-source and available under the **MIT License**.


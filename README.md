# 🪩 Gesture-Controlled Carrom

A modern, touchless twist on the classic tabletop game. Instead of a physical striker, this game uses **Computer Vision** to track your finger movements via webcam, allowing you to flick, aim, and pocket coins in a virtual arena.

---

## 🚀 Features
*   **Touchless Gameplay:** Play entirely through hand gestures—no mouse or keyboard required.
*   **Real-time Finger Tracking:** High-speed detection of fingertips for precise aiming.
*   **Physics Engine:** Realistic collisions, friction, and momentum for the coins and striker.
*   **Dynamic UI:** On-screen overlays that show your "power meter" and trajectory based on your hand position.

---

## 🛠️ How It Works
The game utilizes a camera feed to map your hand coordinates onto the game board. 

1.  **Detection:** The system identifies your hand landmarks (specifically the index finger and thumb).
2.  **Gesture Mapping:** 
    *   **Aiming:** Moving your index finger side-to-side positions the striker.
    *   **The "Flick":** A rapid forward movement or a pinch-and-release gesture triggers the strike.
3.  **Physics Simulation:** The velocity of your gesture is converted into a force vector applied to the striker.



---

## 💻 Installation

### Prerequisites
*   Python 3.8+
*   A functional Webcam
*   Required Libraries:
    ```bash
    pip install opencv-python mediapipe numpy pygame
    ```

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/bhattacharyaananya/gesture-carrom.git](https://github.com/bhattacharyaananya/gesture-carrom.git)
    ```
2.  Navigate to the project folder:
    ```bash
    cd gesture-carrom
    ```
3.  Run the application:
    ```bash
    python carrom.py
    ```

---

## 🎮 Controls
| Action | Gesture |
| :--- | :--- |
| **Move Striker** | Move your hand horizontally across the frame. |
| **Set Power** | The distance between your thumb and index finger (or gesture speed). |
| **Strike** | A quick "flicking" motion toward the camera or a specific trigger gesture. |
| **Reset Board** | Show a "Palm Open" gesture for 2 seconds. |

---

## 💡 Tips for Best Performance
*   **Lighting:** Ensure your room is well-lit so the camera can clearly distinguish your fingers from the background.
*   **Background:** Try to play against a plain background to reduce "noise" in the detection.
*   **Distance:** Stay approximately 2–3 feet away from the camera for the best tracking scale.

---

## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments
*   **MediaPipe:** For the robust hand-tracking framework.
*   **Pygame:** For the 2D physics and rendering engine.

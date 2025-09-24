# Cube — Gesture-Controlled 3D Cube

An interactive 3D cube application that uses hand gesture recognition to control cube rendering and movement in real time. Built with Python, OpenCV, MediaPipe, and Pygame.

## Features
- Interactive 3D cube rendering (`cube_render.py`)
- Hand gesture recognition and mapping to cube controls (`cube_gesture.py`)
- Core logic and program entry point (`cng.py`)
- Dependencies listed in `requirements.txt`

## Project Structure
CUBE/
│
├── cng.py
├── cube_render.py
├── cube_gesture.py
├── requirements.txt

## How It Works

This project combines computer vision with machine learning to create an interactive 3D environment:

1. **Hand Tracking and Keypoint Detection**  
   Using MediaPipe, the system detects a hand in the webcam feed and extracts 21 key landmarks (finger tips, joints, palm base). These landmarks are normalized and serve as the raw input data.

2. **Feature Extraction**  
   The raw landmark coordinates are transformed into meaningful features:
   - Finger positions relative to the palm
   - Motion history (point history of recent frames)
   - Velocity and direction of movement

3. **Machine Learning Classification**  
   Two lightweight ML models are used:
   - **Keypoint Classifier**: A simple neural network trained on labeled keypoint data to classify static gestures (open hand, closed fist, pointer).  
   - **Point History Classifier**: A second classifier trained on sequences of past positions to detect dynamic gestures (swipes, drags).  

   Both models are trained beforehand on small, curated datasets of hand poses and movement traces. During runtime, predictions from both classifiers are combined to determine the current gesture.

4. **Gesture → Action Mapping**  
   The predicted gesture is then translated into cube controls:
   - **Open hand** → Pause cube rotation  
   - **Fast swipe left/right/up/down** → Spin cube in that direction  
   - **Pointer finger** → Drag cube across the screen  

   This mapping logic is defined in `cube_gesture.py`.

5. **Cube Rendering**  
   The cube itself is drawn in real time by `cube_render.py` using Pygame’s 2D projection of 3D coordinates. Rotations, translations, and updates are handled frame by frame based on the gesture commands.

## Installation
1. Clone the repository

2. Create and activate a virtual environment (recommended):
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## Usage
Run the main program:
   python cng.py

Controls:
- Open hand → pause cube rotation  
- Fast swipe → spin cube in that direction  
- Pointer movement → drag the cube
- Two open hands ->  Zoom in and out depending on distance between palms



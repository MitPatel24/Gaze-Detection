# 👀 Alert System Based on Gaze Detection 👀

## 📜 **Project Description**
The Gaze Detection Project leverages computer vision techniques using OpenCV and MediaPipe to implement face and gaze detection capabilities. The project includes two primary functionalities:

*  **Face Mesh Detection:** A detailed facial landmark detection system to identify and visualize key facial features such as lips, eyes, eyebrows, and the face outline.
*  **Iris Tracking and Gaze Detection:** A system to track eye movements, calculate blink ratios, and determine gaze direction (left, right, or center). Audio feedback is provided when the gaze is outside predefined boundaries.

## 📂 **Features**  
1) **Face Mesh Detection:** 
- Detect and highlight various facial landmarks.
- Visualize the facial features using translucent overlays on the camera feed.Real-time video processing from two camera feeds.
- Real-time FPS calculation and display.
2) **Iris Tracking and Gaze Detection:**
- Detect and track the iris for gaze direction.
- Calculate blink ratio to identify blinks.
- Real-time audio alerts when gaze moves outside boundaries.
- Video recording functionality for gaze tracking sessions.
3) **Utilities:**
- Custom functions for drawing translucent polygons.
- Tools for calculating Euclidean distances and ratios.
---


## 🚀 **Getting Started**

### **Prerequisites**   
* Ensure the following libraries are installed with the specified versions: 

    ```bash
        python 3.8 or above
        opencv-python==4.10.0.84
        numpy==1.26.4
        mediapipe==0.10.9
        pygame


### **Installation**
Follow these steps to set up the project locally:

1. Clone the repository:
    ```bash
    git clone git@github.com:MitPatel24/Gaze-Detection.git
    cd <working-directory>

2. Install dependencies:
    ```bash
    pip install opencv-python mediapipe numpy pygame

3. Place the required audio files (e.g., Outside_Beep.wav) in the project directory.

## **Contributors**
- Mitkumar Patel -  [MitPatel24](https://github.com/MitPatel24)











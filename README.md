# Drowsiness Detection System using DLib

This project uses facial landmarks to monitor drowsiness in real-time using the Eye Aspect Ratio (EAR) to prompts/alertss whether a person is awake or drowsy.

## 📦 Features

* Real-time video feed processing using OpenCV
* DLib facial landmark detection (68 points)
* EAR-based thresholding to detect eye closure
* Works on a customizable thresholding.

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the application**

   ```bash
   python main.py
   ```

   *The system will open your camera feed and begin monitoring for drowsiness immediately.*

## 📂 Model File

The shape predictor file `shape_predictor_68_face_landmarks.dat` can be fetched from Dlib's official site too.

## 📚 Reference

This implementation is inspired by the concepts presented in:

* Soukupová, T., & Čech, J. (2016). **Real-Time Eye Blink Detection using Facial Landmarks**
  [Read paper (PDF)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
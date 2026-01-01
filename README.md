# 🏊‍♂️ Real-Time Drowning Detection using Deep Learning

**Comparative Analysis of CNN and KAN Models for Real-Time Drowning Detection**

This repository contains the implementation, experiments, and results for a university assignment under **BMCS2133 – Image Processing**, focusing on the development of a **real-time drowning detection system** using computer vision and deep learning techniques.

The project compares **Convolutional Neural Networks (CNN)** and **Kolmogorov–Arnold Networks (KAN)** for classifying human behaviour in water, integrated with **YOLOv8s** for real-time human detection and tracking.

---

## 📌 Project Overview

Drowning is a silent and time-critical emergency that is often difficult to detect through manual surveillance alone. This project aims to enhance aquatic safety by developing an **automated real-time monitoring system** capable of identifying drowning behaviour from visual data.

The system:
- Detects humans in water using **YOLOv8s**
- Classifies detected actions as **swimming**, **treading water**, or **drowning**
- Triggers alerts when sustained drowning behaviour is detected
- Generates automatic PDF incident reports for review

---

## 🎯 Objectives

The main objectives of this project are:

1. To compare the performance of **CNN** and **KAN** models using accuracy, precision, recall, and F1-score
2. To evaluate the impact of **RGB vs Grayscale** inputs on model performance
3. To analyse **training time and inference speed** for real-time suitability
4. To assess the **practical feasibility of KAN** for real-time drowning detection

---

## 🧠 Models & Technologies Used

### Deep Learning Models
- **CNN (Custom Architecture)** – Baseline classification model
- **KAN (Convolutional KAN)** – Emerging architecture with learnable activation functions
- **YOLOv8s** – Real-time human detection and tracking

### Frameworks & Tools
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- NumPy, Pandas, Matplotlib

---

## 📂 Dataset

- **Source**:  
  [Wang-Kaikai Drowning Detection Dataset](https://github.com/Wang-Kaikai/drowning-detection-dataset)

- **Classes**:
  - `swimming`
  - `tread_water`
  - `drowning`

- **Format**:
  - Images: `.jpg`
  - Annotations: YOLO format (`.txt`)

- **Preprocessing**:
  - Class balancing via oversampling
  - Data augmentation (flip, rotation, blur, colour jitter)
  - Normalisation
  - Resized to **32×32** for fair comparison due to hardware constraints

---

## 🏗️ System Architecture

```
Input (Image / Video / Live Camera)
            ↓
YOLOv8s Detection
            ↓
Human Tracking & Cropping
            ↓
CNN / KAN Classification
            ↓
Alert System & Reporting
```

---

## 📊 Results Summary

| Model | Input | Accuracy | Training Time (10 epochs) | Inference Speed |
|------|------|---------|----------------------------|----------------|
| CNN | RGB | 96.10% | ~5.3 minutes | ~24 ms/frame |
| KAN | RGB | **97.37%** | ~83 minutes | ~175 ms/frame |

### Key Observations
- RGB inputs outperform grayscale for both models
- KAN achieves slightly higher accuracy
- CNN is **significantly faster** and more suitable for real-time deployment
- KAN incurs heavy computational overhead during inference

---

## 🖥️ Streamlit Application Features

- Image, video, and live camera input
- Model selection (CNN / KAN)
- Adjustable confidence thresholds
- Real-time bounding box visualisation
  - 🟢 Swimming
  - 🟡 Treading Water
  - 🔴 Drowning
- Audible and visual alerts
- Automatic **PDF incident report generation**

---

## ⚠️ Limitations

- Low image resolution (32×32) due to GPU memory constraints
- Frame-by-frame classification without temporal context
- Confusion between *treading water* and *drowning*
- KAN model not practical for high-FPS real-time use in current form

---

## 🔮 Future Work

- Incorporate **temporal modelling** (LSTM, 3D-CNN, Video Transformers)
- Train on **higher-resolution and more diverse datasets**
- Optimise KAN inference speed or explore hybrid architectures
- Apply model compression (quantisation, pruning)
- Real-world deployment and validation in swimming pool environments

---

## 👥 Team Members

- **Joash Voon Dirui Joash** – CNN Implementation  
- **Tan Hoong Guan** – KAN Implementation  

---

## 📜 License

This repository is intended for **academic and educational purposes only**.  
Please refer to individual libraries (YOLOv8, PyTorch, KAN) for their respective licenses.


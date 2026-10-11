# 🌾 Smart Agriculture & Precision Irrigation System

An end-to-end AI-powered agricultural monitoring and smart irrigation platform integrating real-time IoT sensor telemetry, weather forecasting data, multi-model deep learning prediction, and computer vision-based weed detection.

---

## 🚀 Key Features

* **🏠 Interactive Dashboard (`dashboard.py`):** Built with Streamlit for live monitoring, sensor input adjustments, and model comparisons.
* **🤖 Multi-Model Irrigation AI:** Compares multiple trained architectures (`Improved LSTM`, `TCN`, `Transformer`, and `Random Forest`) to predict optimal irrigation schedules.
* **🌿 Computer Vision Weed Detection:** Utilizes YOLO object detection models with automatic fallback weights for real-time crop and weed monitoring.
* **🌦️ Environmental Integration:** Incorporates real-time meteorological data and soil metrics (`ETo`, precipitation, soil sample measurements).

---

## 📁 Repository Structure

```text
smart_irrigation_system/
│
├── data/                    # Raw telemetry, weather records, and sensor datasets
├── training/                # Model training scripts, utilities, and pipelines
│   ├── models/              # Saved model weights (.pt, .pkl) & metrics (.json)
│   └── processed/           # Cleaned train/test data splits (.csv)
│
├── dashboard.py             # Main Streamlit full-stack web application
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation

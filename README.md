# VisionAI: VisionAI-End-to-End-Computer-Vision-and-Data-Analytics

An end-to-end computer vision and data analytics pipeline designed for real-time object detection, automated logging, and dynamic operational monitoring.

## 🚀 Architecture & Tech Stack
- **Core Engine:** PyTorch, YOLOv8
- **Backend API:** FastAPI
- **Database & Logging:** SQL Server (T-SQL)
- **Frontend / Dashboard:** Streamlit
- **Containerization & Deployment:** Docker

## 📌 Project Overview
VisionAI is built to bridge the gap between heavy computer vision inference and real-time enterprise data processing. The system captures video/image streams, performs high-speed object detection via YOLOv8, logs operational metrics directly into an SQL Server database through a robust FastAPI backend, and renders dynamic controls via a Streamlit dashboard.

## 📂 Project Structure
```text
VisionAI-Core-Analytics-Engine/
│
├── src/
│   ├── detector.py      # YOLOv8 core inference module
│   ├── database.py      # SQL Server connection & logging handlers
│   └── api.py           # FastAPI endpoints
│
├── dashboard/
│   └── app.py           # Streamlit dynamic monitoring UI
│
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
└── README.md

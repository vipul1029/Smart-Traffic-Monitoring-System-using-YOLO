# 🚦 Smart Traffic Monitoring System using YOLO

## 📌 Overview
This project is an AI-powered traffic monitoring system that detects and tracks vehicles in real-time using YOLOv8. It is designed for smart city applications such as traffic analysis, congestion monitoring, and automated surveillance.

The system processes video input, identifies vehicles, and displays output with bounding boxes and detection insights.

---

## ✨ Features
- 🚗 Real-time vehicle detection using YOLOv8  
- 📊 Traffic analysis from video streams  
- 📍 Line crossing detection (vehicle counting)  
- 🧠 High accuracy with deep learning model  
- 🖼️ Output visualization with bounding boxes  
- ⚡ Fast and efficient processing  

---

## 🛠️ Tech Stack
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  

---
```
## 📂 Project Structure

Smart-Traffic-Monitoring-System-using-YOLO/
│
├── vehicle_detection.py # Main detection script
├── line_cross.py # Line crossing / vehicle counting
├── yolov8n.pt # Pre-trained YOLO model
├── vehicle_output.jpg # Sample output
└── README.md
```

---

## ▶️ Usage
Run vehicle detection:

python vehicle_detection.py


Run line crossing detection:

python line_cross.py


---

## 📸 Output
Sample output showing detected vehicles with bounding boxes.

---

## 🧠 How It Works
1. YOLOv8 model detects vehicles in each frame  
2. OpenCV processes the video stream  
3. Bounding boxes are drawn around detected vehicles  
4. Line-crossing logic counts vehicles  
5. Output is displayed in real-time  

---

## 📈 Future Improvements
- Add DeepSORT tracking for unique vehicle IDs  
- Build a web dashboard (FastAPI + Next.js)  
- Add analytics (traffic trends, charts)  
- Implement real-time alerts system  
- Deploy on cloud with live camera feeds  

---

## 👨‍💻 Author
Vipul Kumar  
Portfolio: https://portfolio-vipul1007s-projects.vercel.app/  
GitHub: https://github.com/vipul1029  

---

## ⭐ Contributing
Contributions are welcome! Feel free to fork and improve the project.

---

## 📄 License
This project is open-source and available under the MIT License.

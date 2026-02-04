# AI-Powered Video Analytics with OpenVINO

## 🚀 Project Overview
This project is a real-time Customer Experience Analytics System designed for physical retail environments. It leverages computer vision to track visitor footfall and analyze customer sentiment (emotion) from live video feeds.

By combining YOLOv8 for person detection and Intel OpenVINO™ for emotion recognition, the system provides actionable business insights—such as "Total Visitors" and "Customer Happiness Scores"—on a professional, dark-mode dashboard.

## 🔥 Key Features
- **Live Person Tracking**: Detects and tracks unique visitors in real-time using YOLOv8.
- **Emotion Recognition**: Analyzes facial expressions (Happy, Neutral, Sad, etc.) using OpenVINO.
- **Interactive Dashboard**: A professional web interface with live video, active tables, and dynamic charts.
- **High Performance**: Optimized for standard CPUs using Intel's OpenVINO toolkit.
- **Privacy Focused**: Processes video locally without sending streams to the cloud.

## 🏗️ Tech Stack
- **Programming Language**: Python 3.9+
- **Computer Vision**: Ultralytics YOLOv8, OpenCV, Intel OpenVINO™
- **Backend**: Flask (Python Web Server), Flask-Sock (WebSockets)
- **Frontend**: HTML5, CSS3 (Dark Theme), JavaScript, Chart.js
- **Hardware Support**: Optimized for CPU (Intel Core/Xeon), compatible with GPU.

## 📂 Project Structure
```plaintext
📁 AI-Powered-Video-Analytics/
├── 📁 models/                     # AI Models storage
│   ├── emotions-recognition...xml # OpenVINO Emotion Model
│   ├── emotions-recognition...bin # Model Weights
│   └── yolov8n.pt                 # YOLOv8 Person Detection Model
├── 📁 templates/
│   └── index.html                 # Professional Dashboard UI
├── 📁 venv/                       # Virtual Environment (Ignored by Git)
├── final_demo.py                  # MAIN APPLICATION (Run this file)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git configuration
└── README.md                      # Project documentation
```

## ⚙️ Installation
```bash
#1. Clone the repository
git clone https://github.com/SubhankarChand/AI-Powered-Video-Analytics-with-OpenVINO.git
cd AI-Powered-Video-Analytics-with-OpenVINO

#2. Set Up Virtual Environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

#3. Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application
```bash
#1. Start the System Run the main demo script which starts the Flask server and AI engine:
python final_demo.py

#2. Access the Dashboard Open your web browser and go to:
[python src/infer.py --input "customer_feedback.txt"](http://localhost:5001)

#3. Stop the App Press Ctrl + C in your terminal to shut down the server.

```

## 📌 Future Improvements
- Multi-Camera Support: Scale the system to handle multiple RTSP feeds.
- Heatmap Generation: Visualize high-traffic zones in the store.
- Demographic Analysis: Add Age and Gender detection models.
- Cloud Sync: Push daily summary reports to a cloud database (AWS/Firebase).

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have ideas for optimization or new features.

## 📜 License


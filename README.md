# 👁️ AI-Powered Video Analytics with OpenVINO™

## 🚀 Project Overview
This project is a real-time **Customer Experience Analytics System** designed for physical retail environments. It transforms standard camera feeds into actionable business intelligence by tracking visitor footfall and analyzing customer sentiment (emotions).

By combining **YOLOv8** for person detection and **Intel® OpenVINO™** for high-performance emotion recognition, the system provides sub-second inference on standard CPUs, visualized through a professional, zero-scroll web dashboard.

## 🔥 Key Features
- **Real-Time Visitor Tracking**: Detects and tracks unique visitors using YOLOv8 and centroid tracking logic.
- **Emotion Recognition**: Analyzes facial expressions (Happy, Neutral, Sad, Anger, Surprise) using optimized OpenVINO™ models.
- **SaaS-Style Dashboard**: A modern, single-page UI featuring live video, dynamic Chart.js analytics, and active tracking logs.
- **Edge Optimized**: Specifically tuned for Intel CPUs to ensure high FPS without the need for an expensive dedicated GPU.
- **Automated Alerts**: Integrated SMTP notification system that triggers email alerts when specific sentiments (e.g., Anger) are detected.

## 🏗️ Tech Stack
- **AI/ML Engine**: Ultralytics YOLOv8, Intel® OpenVINO™ Toolkit.
- **Backend**: Flask (Python), Flask-Sock (WebSockets for real-time telemetry).
- **Database**: SQLite3 (Local persistent logging).
- **Frontend**: HTML5, CSS3 (Modern Dark Theme), JavaScript, Chart.js.
- **DevOps**: Python-Dotenv for secure credential management.

## 📂 Project Structure
```plaintext
📁 AI-Powered-Video-Analytics/
├── 📁 models/               # YOLOv8 (.pt) and OpenVINO IR files (.xml, .bin)
├── 📁 templates/            # Dashboard UI (index.html)
├── .env                     # API Keys & Email Credentials (Private)
├── .gitignore               # Git exclusion rules
├── final_demo.py            # MAIN APPLICATION ENTRY POINT
├── retail_analytics.db      # SQLite Database (Auto-generated)
└── README.md                # Project Documentation
```

## ⚙️ Installation & Setup
* 1. Clone the Repository
``` bash
git clone [https://github.com/SubhankarChand/AI-Powered-Video-Analytics-with-OpenVINO.git](https://github.com/SubhankarChand/AI-Powered-Video-Analytics-with-OpenVINO.git)
cd AI-Powered-Video-Analytics-with-OpenVINO
``` 
* 2. Set Up Virtual Environment
``` Bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
3. Install Dependencies
``` Bash
pip install -r requirements.txt
```
4. Configure Environment Variables
Create a .env file in the root directory and add your SMTP credentials to enable email alerts:

Code snippet
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=target_email@gmail.com
🏃‍♂️ Running the Application
Start the Engine: Run the main script to initialize the Flask server and AI models:

```Bash
python final_demo.py
Access the Dashboard: Open your browser and navigate to:
http://localhost:5001
``` 
Stop the App: Press Ctrl + C in the terminal to shut down the server.

## 📌 Future Improvements
Multi-Camera Support: Scale to handle multiple RTSP streams simultaneously.

Heatmap Generation: Visualize high-traffic zones within the retail space.

Demographic Analysis: Implement Age and Gender detection models.

Cloud Integration: Sync daily summary reports to AWS or Firebase.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any optimizations or feature requests.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

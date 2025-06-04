# 🧠 Intelligent Crime Detection Using Artificial Intelligence

## 📌 Project Overview

**Intelligent Crime Detection Using AI** is an advanced surveillance system that detects potential violence in real-time using motion tracking, face detection, and alert mechanisms. Built with Python, OpenCV, and deep learning, the system captures video streams, detects suspicious movement and faces, and generates alerts, while maintaining logs in both CSV and Excel formats.

---

## 🎯 Objective

To build a smart, real-time monitoring system that detects criminal or suspicious behavior using AI techniques, helping reduce the burden on manual surveillance systems and increasing safety through timely alerts and logging.

---

## 🛠️ Features

- 🎥 Real-time **video stream processing**
- 🔍 Accurate **face detection** using DNN-based models
- 🎯 **Motion detection** with frame differencing
- 📢 **Audio alerts** when suspicious activity is detected
- 📝 **CSV log generation** of all events with motion score and alert status
- 📊 **Automatic conversion of logs to Excel**
- 💡 Easy-to-integrate with surveillance systems or smart cities

---

## 🧪 Technologies Used

- **Programming Language:** Python
- **Libraries & Tools:** OpenCV, NumPy, pygame, pandas, imutils, CSV, datetime
- **Models Used:** DNN Face Detection (`res10_300x300_ssd_iter_140000.caffemodel`)
- **IDE/Environment:** VS Code / Jupyter Notebook
- **Alert Mechanism:** Pygame Sound Playback
- **Logging:** CSV and Excel via `pandas`

---

## 📁 Project Structure
- ├── deploy.prototxt.txt # DNN face detection model config
- ├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detection model
- ├── Beep.wav # Alert sound file
- ├── detection_log.csv # Auto-generated log file
- ├── violence-detector4.py # Main code for detection
- └── README.md # Project documentation



---

## 🚀 How to Run

### ✅ Step 1: Clone the repository

- git clone https://github.com/yourusername/violence-detection-using-artificial-intelligence.git
- cd intelligent-crime-detection-ai

### ✅ Step 2: Install dependencies
pip install opencv-python imutils pygame pandas numpy

### ✅ Step 3: Run the Program
python violence-detector4.py



##📂 Log Generation
- All detections are saved with timestamp, motion score, face count, and alert status in detection_log.csv.
- On program exit, it automatically creates a timestamped .xlsx Excel log file for easy review and audit.



##🧪 Sample Output
- Detects human faces using DNN and OpenCV
- Identifies motion in frame difference (based on pixel intensity)
- Logs events like:
  Timestamp, Motion Score, Face Events, Alert Status
  2025-06-04 14:35:12, 8129, 1, Alert Triggered
- Converts .csv to .xlsx like:
detection_log_2025-06-04_14-35-12.xlsx



## 🚀 Future Enhancements
- 📩 Email/SMS alerts on violence detection
- 🔗 Integration with external face recognition APIs
- ☁️ Cloud upload of event logs
- 🎥 Multi-camera support
- 🤖 Deep learning-based behavior recognition

______________________________________________________________________________________________________________________________________________________________

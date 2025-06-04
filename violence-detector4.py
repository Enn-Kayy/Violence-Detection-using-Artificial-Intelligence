import cv2
import numpy as np
import imutils
import pygame
import os
import csv
from datetime import datetime
import pandas as pd  # Library to handle CSV to Excel conversion

# Initialize pygame mixer for sound
pygame.mixer.init()
alert_sound_path = r'C:\Users\Naman Kapoor\OneDrive\Documents\violence-detection\Beep.wav'
alert_sound = pygame.mixer.Sound(alert_sound_path)

# Load the pre-trained face detection model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Log file path
LOG_FILE_PATH = r'C:\Users\Naman Kapoor\OneDrive\Documents\violence-detection\detection_log.csv'

# Logging function using CSV format with alert status
def log_to_file(motion_score, face_events, alert_status):
    file_exists = os.path.exists(LOG_FILE_PATH)
    
    with open(LOG_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new or empty
        if not file_exists or os.stat(LOG_FILE_PATH).st_size == 0:
            writer.writerow(["Timestamp", "Motion Score", "Face Events", "Alert Status"])
        
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), motion_score, face_events, alert_status])

# Face detection function
def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces

# Main function
def detect_violence(video_source):
    print("📷 Starting camera...")
    
    # Clear previous CSV log before starting
    if os.path.exists(LOG_FILE_PATH):
        open(LOG_FILE_PATH, 'w').close()
        print("🧹 Previous CSV log cleared.")
        
    cap = cv2.VideoCapture(video_source)
    avg_frame = None
    face_detection_count = 0

    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_frame is None:
            avg_frame = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg_frame, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_score = int(np.sum(thresh) / 255)
        face_detection_count = 0

        for contour in contours:
            if cv2.contourArea(contour) < 700:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            roi_color = frame[y:y+h, x:x+w]
            faces = detect_faces(roi_color)

            if len(faces) > 0:
                face_detection_count += 1
                cv2.putText(frame, "Warning: Potential Violence Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Determine alert status and play alert
        if face_detection_count > 0:
            alert_status = "Alert Triggered"
            alert_sound.play()  # Reset after triggering alert
        else:
            alert_status = "No Alert"

        # Log data to CSV
        log_to_file(motion_score, face_detection_count, alert_status)

        # Display the frame
        cv2.imshow("Violence Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("👋 Quit signal received. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert CSV log to Excel
    print("📁 Converting CSV log to Excel...")

    try:
        df = pd.read_csv(LOG_FILE_PATH)

        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(LOG_FILE_PATH))[0]
        folder_path = os.path.dirname(LOG_FILE_PATH)
        excel_file_path = os.path.join(folder_path, f"{base_name}_{timestamp}.xlsx")

        df.to_excel(excel_file_path, index=False)
        print(f"✅ Excel file created at: {excel_file_path}")
    except Exception as e:
        print(f"❌ Failed to convert CSV to Excel: {e}")

# Run the detection
detect_violence(0)

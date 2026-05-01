import cv2
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import time
import os
import tkinter as tk
from tkinter import filedialog

# detect setting
LOCATION = "Perungalathur Bus Stop"
CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 3000
CONFIRM_FRAMES = 3
EMAIL_COOLDOWN = 30

MODEL_PATH = "../model/best.pt"

#  ONLY REQUIRED CLASSES
CRIME_CLASSES = ['violence', 'weapon']

#  COLORS ONLY FOR USED CLASSES
CLASS_COLORS = {
    'violence': (0, 0, 255),   # Red
    'weapon': (255, 0, 0)      # Blue
}

# mail
sender_email = "abisheakabi27@gmail.com"
receiver_email = "abisheakkcsus21083@nct.ac.in"
app_password = "sfubibowftdbrhlb"

last_email_time = 0

def send_email(subject, message, image_path=None):
    global last_email_time

    if time.time() - last_email_time < EMAIL_COOLDOWN:
        return

    try:
        msg = EmailMessage()
        msg.set_content(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
                msg.add_attachment(img_data,
                                   maintype='image',
                                   subtype='jpeg',
                                   filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)

        print(" Email Sent with Attachment")
        last_email_time = time.time()

    except Exception as e:
        print(" Email Error:", e)

# log
def log_event(text):
    os.makedirs("../logs", exist_ok=True)
    with open("../logs/logs.txt", "a") as f:
        f.write(f"{time.ctime()} - {text}\n")

# load model
model = YOLO(MODEL_PATH)

# input
print("\nSelect Input Source:")
print("1. Webcam")
print("2. Upload Video")

choice = input("Enter 1 / 2: ")

if choice == "1":
    cap = cv2.VideoCapture(0)

elif choice == "2":
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    root.destroy()

    if not video_path:
        print("No video selected")
        exit()

    cap = cv2.VideoCapture(video_path)

else:
    print("Invalid choice")
    exit()

if not cap.isOpened():
    print("Error opening video source")
    exit()

# output
os.makedirs("../output/incidents", exist_ok=True)

incident_counter = 0

cv2.namedWindow("Smart Surveillance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Surveillance System", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    annotated = frame.copy()

    incident_detected = False
    detected_labels = []

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area < MIN_BOX_AREA:
                continue

            # FILTER ONLY REQUIRED CLASSES
            if label not in CRIME_CLASSES:
                continue

            incident_detected = True
            detected_labels.append(label)

            color = CLASS_COLORS.get(label, (0,0,255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # confirm incident
    if incident_detected:
        incident_counter += 1
    else:
        incident_counter = 0

    if incident_counter >= CONFIRM_FRAMES:

        cv2.putText(
            annotated,
            "INCIDENT DETECTED",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            3
        )

        filename = f"../output/incidents/incident_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated)

        log_event(f"Incident Detected: {set(detected_labels)}")

        send_email(
            "🚨 ALERT: Crime Detected",
            f"Location: {LOCATION}\nTime: {time.ctime()}\nDetected: {set(detected_labels)}",
            image_path=filename
        )

    cv2.imshow("Smart Surveillance System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
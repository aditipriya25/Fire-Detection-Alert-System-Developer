import cv2
import cvzone
import math
import argparse
import os
import logging
from ultralytics import YOLO
from twilio.rest import Client

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to send SMS
def send_sms():
    try:
        account_sid = 'your_twilio_sid'  # Replace with your Twilio SID
        auth_token = 'your_twilio_auth_token'  # Replace with your Twilio Auth Token
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body='🔥 Fire has been detected by the UAV!',
            from_='your_twilio_number',  # Replace with Twilio number
            to='receiver_number'  # Replace with receiver's phone number
        )

        logging.info(f"SMS sent successfully. SID: {message.sid}")
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")

# Load the YOLO model
model_path = 'best.pt'
if not os.path.exists(model_path):
    logging.error("Model file 'best.pt' not found! Ensure it's in the project directory.")
    exit(1)

model = YOLO(model_path)
classnames = ['fire']
sms_sent = False  # Prevent multiple SMS alerts

# Argument parser for choosing input source
parser = argparse.ArgumentParser(description="Fire Detection using YOLOv8")
parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam will be used.")
args = parser.parse_args()

# Select input source
if args.video:
    if not os.path.exists(args.video):
        logging.error(f"Video file '{args.video}' not found!")
        exit(1)
    cap = cv2.VideoCapture(args.video)
    logging.info(f"Using video file: {args.video}")
else:
    cap = cv2.VideoCapture(0)  # Webcam
    logging.info("Using webcam for real-time detection.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("End of video or failed to capture frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_id = int(box.cls[0])

            if confidence > 50 and class_id == 0:  # Class ID for 'fire'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(frame, f'{classnames[class_id]} {confidence}%', (x1 + 5, y1 + 35),
                                   scale=1.2, thickness=2)

                if not sms_sent:
                    logging.info("🔥 Fire detected! Sending SMS alert...")
                    send_sms()
                    sms_sent = True  # Ensure only one alert is sent

    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

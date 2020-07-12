from imutils.video import VideoStream
from imutils.video import FPS
from twilio.rest import Client
import numpy as np
import threading
import datetime
import imutils
import pickle
import time
import json
import cv2
import os


class Activities:
    def __init__(self):
        # Initialize Twilio Client to send whatsapp message
        self.client = Client()
        # Load the JSON configuration file
        self.conf = json.load(open('conf.json'))

        self.details_json = self.conf['details_json']
        # Load OpenCV’s Caffe-based deep learning face detector model
        print("[EXEC] Loading face detector...")
        self.detector = cv2.dnn.readNetFromCaffe(os.path.sep.join([self.conf["detector"], "deploy.prototxt"]),
                                                 os.path.sep.join([self.conf["detector"],
                                                                   "res10_300x300_ssd_iter_140000.caffemodel"])
                                                 )
        # Load our face embeddings
        print("[EXEC] loading face embeddings...")
        self.embedder = cv2.dnn.readNetFromTorch(self.conf["embedding_model"])

        # Load the recogniser model
        print("[EXEC] loading face recognizer...")
        self.recognizer = pickle.loads(open(self.conf["recognizer"], "rb").read())

        # Load the Label encoder
        self.le = pickle.loads(open(self.conf["le"], "rb").read())

    # save the captured frame in a file
    def store_frame(self, get_name, op_frame, image_counter, timestamp):
        self.details_json[get_name]['total_capture'] = self.details_json[get_name]['total_capture'] + 1
        self.details_json[get_name]["timestamp"] = timestamp
        self.details_json[get_name]["captures"] = image_counter

        p = os.path.sep.join([self.conf["image_path"], "{}_{}_{}.png".format(
            str(get_name).capitalize(), str(timestamp.strftime("%d_%B_%Y_%I:%M%p")).zfill(5),
            str(self.details_json[get_name]['total_capture']))])

        cv2.imwrite(p, op_frame)

    # send whatsapp message to specific number via TWILIO API
    def send_message(self, get_name, timestamp):
        self.client.messages.create(
            body='Last seen: {} \n@ {}'.format(str(get_name).upper(), str(timestamp.strftime("%d %B %Y %I:%M%p"))),
            from_=self.conf['FROM_NUMBER'],
            to=self.conf['TO_NUMBER']
        )


# initialse a activity object
activity = Activities()

# Load videostream
print("[EXEC] starting video stream...")
vs = VideoStream(src=0, framerate=activity.conf['fps']).start()
# Let the camera sensor warm-up
time.sleep(activity.conf["camera_warmup_time"])

frame_avg = None
last_seen = datetime.datetime.now()

while True:
    # Read frames
    frame = vs.read()
    # Current timestamp
    curr_timestamp = datetime.datetime.now()

    # Frame resize
    frame = imutils.resize(frame, width=600)
    # Height and width of the frame
    (h, w) = frame.shape[:2]

    # Pre-process image by Mean subtraction, Resize and scaling by some factor
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    activity.detector.setInput(imageBlob)
    # Detect possible face detection in image with the detector model
    detections = activity.detector.forward()

    # covert frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove high frequency noise
    # This allows us to focus on the “structural” objects of the image.
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # gray frame will be used for motion detection

    # if the average frame is None, initialize it
    if frame_avg is None:
        print("[EXEC] starting background model...")
        frame_avg = gray.copy().astype("float")
        continue

    # weighted mean of previous frames along with the current frame
    cv2.accumulateWeighted(gray, frame_avg, 0.5)
    # subtract the weighted average from the current frame
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(frame_avg))
    # threshold this delta to find regions of our image that
    # contain substantial difference from the background model
    # these regions thus correspond to “motion” in our video stream
    thresh = cv2.threshold(frameDelta, activity.conf["delta_thresh"], 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find over the thresh image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # loop over the contours
    for c in contours:
        # Apply face detection contours are large enough
        # i.e Detect face and apply recognition on motion detection
        if cv2.contourArea(c) < activity.conf["min_area"]:
            continue

        # Now loop over each detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Proceed if detected face confidence is above min_confidence
            if confidence > activity.conf["min_confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # Extract the ROI (region of interest)
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # Make sure ROI is sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # Now we pre-process the our ROI i.e face detected
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                # Use embedder model to extract 128-d face embeddings
                activity.embedder.setInput(faceBlob)
                vec = activity.embedder.forward()

                # Now make predictions based on our recognizer model
                preds = activity.recognizer.predict_proba(vec)[0]
                # store the index prediction maximum probability
                j = np.argmax(preds)
                # store the prediction maximum probability
                proba = preds[j]
                # extract the name of the prediction
                name = activity.le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                ts = curr_timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
                text = "Motion detected"
                cv2.putText(frame, "{}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                if name == "unknown":
                    continue
                # If first frame or person frames captured less than 3
                # store the frame capture to file
                # send whatsapp notification
                elif activity.details_json[name]["captures"] == 0 or activity.details_json[name]["captures"] < \
                        activity.conf["max_captures"]:
                    # start new thread and call store_frame function
                    threading.Thread(
                        target=activity.store_frame(
                            op_frame=frame, image_counter=activity.details_json[name]["captures"] + 1,
                            get_name=name,
                            timestamp=curr_timestamp
                        )
                    ).start()

                    # If whatsapp notfication not sent
                    # Send the last seen person name and timestamp
                    if not activity.details_json[name]['image_sent']:
                        threading.Thread(target=activity.send_message(
                            get_name=name, timestamp=curr_timestamp)
                        ).start()
                        activity.details_json[name]['image_sent'] = True
                # If the person is seen again after certain time then only
                # store the frame capture to file
                # send whatsapp notification
                elif abs(activity.details_json[name]['timestamp'] - curr_timestamp).total_seconds() \
                        >= activity.conf["min_time_gap"]:
                    threading.Thread(
                        target=activity.store_frame(
                            op_frame=frame, get_name=name, image_counter=1, timestamp=curr_timestamp)
                    ).start()
                    threading.Thread(
                        target=activity.send_message(get_name=name, timestamp=curr_timestamp)
                    ).start()

    # show the current videostram or not
    if activity.conf["show_video"]:
        # display the security feed
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            print(activity.details_json)
            break
cv2.destroyAllWindows()
vs.stop()


# -------------After every restart of the local machine execute the below
# only once before the run command--------------------
# export TWILIO_ACCOUNT_SID='ACfc95495fa8cb6cb82bbc3cf77d96a43f'
# export TWILIO_AUTH_TOKEN='596cc9a0cf66d07ce546048870617856'

# ---------------Run Command-----------
# python faceRecognition_notification.py

from util import list_files
import numpy as np
import pickle
import cv2
import os

MIN_CONFIDENCE = 0.55

# Load the OpenCV’s Caffe-based deep learning face detector model
print("[EXEC] Loading face detector model...")
detector = cv2.dnn.readNetFromCaffe("face_detection_model/deploy.prototxt",
                                    "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

# Load the embbeder model to extract a 128-D facial embedding vector
# It contains the OpenCV deep learning Torch embedding model.
print("[EXEC] Loading face recognizer model...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

print("[EXEC] Reading Image Paths.....")
# Discrete each image path into a list
imagePaths = list(list_files(rootPath="my_dataset"))
print(imagePaths)

knownEmbeddings = []
knownNames = []

total = 0

# Iterate over every single image
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {} of {}".format(i + 1, len(imagePaths)))
    # Extract name of the image
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, dsize=(750, 600))
    # Height and Width
    (h, w) = image.shape[:2]

    # Pre-process image by Mean subtraction, Resize and scaling by some factor
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    # Detect possible face detection in image with the detector model
    detections = detector.forward()

    # Proceed if Faces is detected
    if len(detections) > 0:
        # Index detection with highest detected face confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Proceed if detected face confidence is above min_confidence
        if confidence > MIN_CONFIDENCE:
            # Bounding box of face detected
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the ROI (region of interest)
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Make sure ROI is sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Now we pre-process the our ROI i.e face detected
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)

            # Use embedder model to extract 128-d face embeddings
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Append the name and the embeding vector
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print("[EXEC] Collecting {} encodings vectors...".format(total))
# Each encoding/embedding is 128-D vector
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Save the embedding to a file
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# -------------Run Command-----------------
# python face_extract_embedding.py

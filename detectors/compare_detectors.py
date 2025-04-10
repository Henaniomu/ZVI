import os
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt


HAAR_PATH = "models/haarcascade_frontalface_default.xml"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
DNN_PROTO = "models/deploy.prototxt"

haar_cascade = cv2.CascadeClassifier(HAAR_PATH)
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)


def detect_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    end = time.time()
    return faces, end - start


def detect_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    start = time.time()
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    end = time.time()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            boxes.append(box.astype("int"))
    return boxes, end - start


def process_image_streamlit(image_path, save=True):
    img = cv2.imread(image_path)
    if img is None:
        return None

    haar_faces, haar_time = detect_haar(img)
    dnn_faces, dnn_time = detect_dnn(img)

    haar_img = img.copy()
    for (x, y, w, h) in haar_faces:
        cv2.rectangle(haar_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    dnn_img = img.copy()
    for (x1, y1, x2, y2) in dnn_faces:
        cv2.rectangle(dnn_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return {
        "filename": os.path.basename(image_path),
        "haar_img": cv2.cvtColor(haar_img, cv2.COLOR_BGR2RGB),
        "dnn_img": cv2.cvtColor(dnn_img, cv2.COLOR_BGR2RGB),
        "haar_faces": len(haar_faces),
        "haar_time": haar_time,
        "dnn_faces": len(dnn_faces),
        "dnn_time": dnn_time
    }


def run_face_detection(file_paths):
    results = []
    for path in file_paths:
        result = process_image_streamlit(path, save=False)
        if result:
            results.append(result)
    return results


def plot_detection_stats(results):
    df = pd.DataFrame(results)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df["filename"], df["haar_faces"], label="Haar", alpha=0.6)
    ax1.bar(df["filename"], df["dnn_faces"], label="DNN", alpha=0.6)
    ax1.set_title("Number of Faces Detected")
    ax1.set_ylabel("Faces")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(df["filename"], df["haar_time"], label="Haar Time", alpha=0.6)
    ax2.bar(df["filename"], df["dnn_time"], label="DNN Time", alpha=0.6)
    ax2.set_title("Detection Time per Image")
    ax2.set_ylabel("Time (s)")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig1, fig2

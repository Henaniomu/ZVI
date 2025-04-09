import os
import cv2
import time
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import imageio


# Пути к моделям
HAAR_PATH = "models/haarcascade_frontalface_default.xml"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
DNN_PROTO = "models/deploy.prototxt"

# Загрузка моделей
haar_cascade = cv2.CascadeClassifier(HAAR_PATH)
dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)


# Haar функция
def detect_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    end = time.time()
    return faces, end - start


# DNN функция
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


# Универсальный обработчик
def process_image(image_path, save=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't read image {image_path}")
        return None

    haar_faces, haar_time = detect_haar(img)
    dnn_faces, dnn_time = detect_dnn(img)

    # Отрисовка и сохранение
    if save:
        os.makedirs("results_haar", exist_ok=True)
        os.makedirs("results_dnn", exist_ok=True)

        haar_img = img.copy()
        for (x, y, w, h) in haar_faces:
            cv2.rectangle(haar_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        dnn_img = img.copy()
        for (x1, y1, x2, y2) in dnn_faces:
            cv2.rectangle(dnn_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite(f"results_haar/haar_{os.path.basename(image_path)}", haar_img)
        cv2.imwrite(f"results_dnn/dnn_{os.path.basename(image_path)}", dnn_img)

    print(f"\n{os.path.basename(image_path)}:")
    print(f"Haar: {len(haar_faces)} face(s) in {haar_time:.4f} sec")
    print(f"DNN : {len(dnn_faces)} face(s) in {dnn_time:.4f} sec")

    return {
        "image": os.path.basename(image_path),
        "haar_faces": len(haar_faces),
        "haar_time": haar_time,
        "dnn_faces": len(dnn_faces),
        "dnn_time": dnn_time
    }


# Обработка папки
def process_folder(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    all_stats = []

    for img_file in tqdm(images):
        result = process_image(os.path.join(folder, img_file))
        if result:
            all_stats.append(result)

    # Log save
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(all_stats)
    df.to_csv("logs/results.csv", index=False)
    print("\nCSV log saved to logs/results.csv")

    # Graphs gen
    plot_stats(df)


def plot_stats(df):
    os.makedirs("plots", exist_ok=True)

    # График: количество лиц
    plt.figure(figsize=(10, 5))
    plt.bar(df["image"], df["haar_faces"], label="Haar", alpha=0.7)
    plt.bar(df["image"], df["dnn_faces"], label="DNN", alpha=0.7)
    plt.title("Number of Faces Detected")
    plt.xlabel("Image")
    plt.ylabel("Faces")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/faces_detected.png")
    print("Saved plot: plots/faces_detected.png")

    # График: время обработки
    plt.figure(figsize=(10, 5))
    plt.bar(df["image"], df["haar_time"], label="Haar Time", alpha=0.7)
    plt.bar(df["image"], df["dnn_time"], label="DNN Time", alpha=0.7)
    plt.title("Detection Time (seconds)")
    plt.xlabel("Image")
    plt.ylabel("Time")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/detection_time.png")
    print("Saved plot: plots/detection_time.png")


def run_face_detection():
    folder = "images"
    process_folder(folder)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Haar and DNN face detection")
    parser.add_argument("--image", type=str, help="Path to one image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--nosave", action="store_true", help="Don't save results")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, save=not args.nosave)
    elif args.folder:
        process_folder(args.folder)
    else:
        print("Use --image or --folder")

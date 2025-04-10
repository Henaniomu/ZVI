import cv2
import time
import os
import argparse
from tqdm import tqdm

# Model and conf
MODEL_FILE = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "models/deploy.prototxt"

# DNN load
net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

def detect_faces_dnn(image_path, save=True, show=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    start = time.time()
    net.setInput(blob)
    detections = net.forward()
    end = time.time()

    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if save:
        os.makedirs("results_dnn", exist_ok=True)
        output_path = os.path.join("results_dnn", f"dnn_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, image)

    if show:
        cv2.imshow("DNN Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "file": os.path.basename(image_path),
        "faces": face_count,
        "time": end - start
    }


def process_folder(folder_path, save=True, show=False):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    stats = []

    print(f"Processing {len(image_files)} images with OpenCV DNN...\n")

    for file in tqdm(image_files):
        full_path = os.path.join(folder_path, file)
        result = detect_faces_dnn(full_path, save=save, show=show)
        if result:
            stats.append(result)

    print("\nDNN Summary:")
    for entry in stats:
        print(f"- {entry['file']}: {entry['faces']} face(s), {entry['time']:.4f} sec")

    if stats:
        avg_time = sum(e['time'] for e in stats) / len(stats)
        total_faces = sum(e['faces'] for e in stats)
        print(f"\nProcessed {len(stats)} images, total faces: {total_faces}, average time: {avg_time:.4f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using OpenCV DNN.")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder with images")
    parser.add_argument("--nosave", action="store_true", help="Do not save result images")
    parser.add_argument("--show", action="store_true", help="Show images with detection boxes")

    args = parser.parse_args()

    if args.image:
        res = detect_faces_dnn(args.image, save=not args.nosave, show=args.show)
        if res:
            print(f"\nDetected {res['faces']} face(s) in {res['file']} ({res['time']:.4f} sec)")
    elif args.folder:
        process_folder(args.folder, save=not args.nosave, show=args.show)
    else:
        print("Please provide --image or --folder")

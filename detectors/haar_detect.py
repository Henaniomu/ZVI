import cv2
import os
import time
import argparse
from tqdm import tqdm

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def detect_faces(image_path, save=True, show=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    end = time.time()

    detection_time = end - start
    face_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save:
        result_path = os.path.join("results", f"detected_{os.path.basename(image_path)}")
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(result_path, img)

    if show:
        cv2.imshow("Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "file": os.path.basename(image_path),
        "faces": face_count,
        "time": detection_time
    }


def process_folder(folder_path, save=True, show=False):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    stats = []

    print(f"Processing {len(image_files)} images in '{folder_path}'...\n")

    for file in tqdm(image_files):
        full_path = os.path.join(folder_path, file)
        result = detect_faces(full_path, save=save, show=show)
        if result:
            stats.append(result)

    print("\nDetection Summary:")
    for entry in stats:
        print(f"- {entry['file']}: {entry['faces']} face(s), {entry['time']:.4f} sec")

    if stats:
        avg_time = sum(e['time'] for e in stats) / len(stats)
        total_faces = sum(e['faces'] for e in stats)
        print(f"\nProcessed {len(stats)} images, total faces: {total_faces}, average time: {avg_time:.4f} sec")


def run_face_detection():
    folder = input("Enter the folder path with images: ").strip()
    if not os.path.exists(folder):
        print("Folder doesn't exist.")
        return
    process_folder(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using Haar Cascade.")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder with images")
    parser.add_argument("--nosave", action="store_true", help="Do not save result images")
    parser.add_argument("--show", action="store_true", help="Show images with detection boxes")

    args = parser.parse_args()

    if args.image:
        res = detect_faces(args.image, save=not args.nosave, show=args.show)
        if res:
            print(f"\nDetected {res['faces']} face(s) in {res['file']} ({res['time']:.4f} sec)")
    elif args.folder:
        process_folder(args.folder, save=not args.nosave, show=args.show)
    else:
        print("Please provide --image or --folder")

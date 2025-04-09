import dlib
import cv2
import os

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

IMAGE_FOLDER = "images"
RESULT_FOLDER = "results/landmarks"
os.makedirs(RESULT_FOLDER, exist_ok=True)

def run_landmarks():
    print("\nStarting face landmarks detection (Dlib)...")

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in images:
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image: {img_file}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        if not faces:
            print(f"No face detected in {img_file}")
            continue

        for face in faces:
            landmarks = shape_predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        result_path = os.path.join(RESULT_FOLDER, f"landmarks_{img_file}")
        cv2.imwrite(result_path, image)
        print(f"Landmarks saved to {result_path}")

    print("\nAll images processed with Dlib.\n")

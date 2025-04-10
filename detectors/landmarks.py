import dlib
import cv2
import os

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

def run_landmarks(file_paths):
    results = []

    for path in file_paths:
        filename = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        if not faces:
            continue

        for face in faces:
            landmarks = shape_predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        results.append({
            "filename": filename,
            "output_img": cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        })

    return results

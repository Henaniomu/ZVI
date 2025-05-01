import dlib
import numpy as np
import cv2

face_detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def get_face_embedding(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) == 0:
        return None

    shape = sp(gray, faces[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def compare_faces_dlib(path1, path2):
    emb1 = get_face_embedding(path1)
    emb2 = get_face_embedding(path2)

    if emb1 is None or emb2 is None:
        return None

    distance = np.linalg.norm(emb1 - emb2)
    if distance < 0.6:
        similarity = 100 - (distance / 0.6) * 20  # 80–100%
    elif distance < 1.0:
        similarity = 80 - ((distance - 0.6) / 0.4) * 50  # 30–80%
    else:
        similarity = max(0, 30 - ((distance - 1.0) * 30))  # 0–30%

    similarity = round(similarity, 2)
    return similarity


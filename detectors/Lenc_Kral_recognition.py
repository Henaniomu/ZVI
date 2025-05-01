import cv2
import dlib
import numpy as np

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
sift = cv2.SIFT_create()

def align_face(gray, landmarks):
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)

    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    aligned = cv2.warpAffine(gray, rot_mat, gray.shape[::-1])
    return aligned

def get_face_descriptors(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if not faces:
        return None

    landmarks = shape_predictor(gray, faces[0])
    aligned_gray = align_face(gray, landmarks)
    
    mask = np.zeros_like(aligned_gray)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)]
    cv2.fillConvexPoly(mask, np.array(points), 255)

    keypoints, descriptors = sift.detectAndCompute(aligned_gray, mask)
    return descriptors

def enhanced_lenc_kral_similarity(desc1, desc2, top_k=20):
    if desc1 is None or desc2 is None:
        return 0
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)[:top_k]

    similarity = sum([1 / (match.distance + 1e-6) for match in matches]) / top_k
    return similarity

def compare_faces_lenc_kral_enhanced(path1, path2):
    desc1 = get_face_descriptors(path1)
    desc2 = get_face_descriptors(path2)

    similarity = enhanced_lenc_kral_similarity(desc1, desc2)
    return similarity

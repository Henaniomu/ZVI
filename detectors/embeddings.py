import dlib
import cv2
import numpy as np
import os

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

PATCH_RADIUS = 16

# Indices for key face parts: eyes, eyebrows, nose, mouth
important_indices = list(range(17, 36)) + list(range(36, 48)) + list(range(48, 68))

hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16),
                        _blockStride=(8, 8), _cellSize=(8, 8),
                        _nbins=9)


def align_face(image, landmarks):
    left_eye = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)], axis=0)
    right_eye = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)], axis=0)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    return aligned_image


def extract_landmark_hog_descriptors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if not faces:
        print(f"No face detected in {image_path}")
        return None

    landmarks = shape_predictor(gray, faces[0])
    image = align_face(image, landmarks)
    landmarks = shape_predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), faces[0])

    descriptors = []
    for i in important_indices:
        x, y = landmarks.part(i).x, landmarks.part(i).y
        x1 = max(x - PATCH_RADIUS, 0)
        y1 = max(y - PATCH_RADIUS, 0)
        x2 = min(x + PATCH_RADIUS, image.shape[1])
        y2 = min(y + PATCH_RADIUS, image.shape[0])
        patch = image[y1:y2, x1:x2]

        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_patch, (64, 64))
        hog_desc = hog.compute(resized)
        descriptors.append(hog_desc.flatten())

    return descriptors


def compare_hog_descriptors(desc1, desc2):
    similarities = []
    for h1, h2 in zip(desc1, desc2):
        if np.linalg.norm(h1) == 0 or np.linalg.norm(h2) == 0:
            similarities.append(0.0)
            continue
        sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
        similarities.append(sim)
    return similarities


def run_embedding_compare():
    print("\nLandmark-based face comparison using HOG features")
    path1 = input("Enter path to first image: ").strip()
    path2 = input("Enter path to second image: ").strip()

    desc1 = extract_landmark_hog_descriptors(path1)
    desc2 = extract_landmark_hog_descriptors(path2)

    if desc1 is None or desc2 is None:
        print("Could not process one or both images.")
        return

    similarities = compare_hog_descriptors(desc1, desc2)
    avg_similarity = np.mean(similarities)

    print(f"\nAverage cosine similarity (HOG-based): {avg_similarity:.4f}")
    percentage = avg_similarity * 100
    print(f"Similarity score: {percentage:.2f}%")

    if percentage > 85:
        print("Faces are very likely the same person.")
    elif percentage > 70:
        print("Faces might be the same person.")
    else:
        print("Faces are likely different.")

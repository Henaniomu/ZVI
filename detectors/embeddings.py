import cv2
import numpy as np
import os

# OpenCV DNN model for face detection and embeddings
MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONFIG = "models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(CONFIG, MODEL)

def extract_face_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (100, 100))  # fix size
            embedding = face_resized.flatten() / 255.0   # normalization
            return embedding

    print(f"No face detected in {image_path}")
    return None

def run_embedding_compare():
    print("\nFace Embedding Comparison")
    path1 = input("Enter path to first image: ").strip()
    path2 = input("Enter path to second image: ").strip()

    emb1 = extract_face_embedding(path1)
    emb2 = extract_face_embedding(path2)

    if emb1 is None or emb2 is None:
        print("Could not extract embeddings from one or both images.")
        return

    # cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    similarity = np.clip(similarity, -1, 1)
    percentage = round(similarity * 100, 2)

    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Similarity score: {percentage}%")

    if percentage > 85:
        print("Faces are likely the same person.")
    elif percentage > 60:
        print("Faces might be the same person.")
    else:
        print("Faces are likely different.")


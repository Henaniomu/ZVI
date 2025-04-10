import streamlit as st
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt

from detectors.compare_detectors import run_face_detection, plot_detection_stats
from detectors.landmarks import run_landmarks
from detectors.embeddings import extract_landmark_hog_descriptors, compare_hog_descriptors


st.set_page_config(page_title="Face Processing Utility", layout="centered")
st.title("🧠 Face Processing Utility")

option = st.selectbox(
    "Choose experiment:",
    (
        "Face Detection (Haar + DNN)",
        "Face Landmarks (Dlib)",
        "Face Comparison (Landmark Embeddings)"
    )
)

# ========== FACE COMPARISON ==========
if option == "Face Comparison (Landmark Embeddings)":
    st.subheader("Select two images for comparison")
    uploaded_file1 = st.file_uploader("First image", type=["jpg", "jpeg", "png"], key="file1")
    uploaded_file2 = st.file_uploader("Second image", type=["jpg", "jpeg", "png"], key="file2")

    if uploaded_file1 and uploaded_file2:
        path1 = os.path.join("temp1.jpg")
        path2 = os.path.join("temp2.jpg")
        with open(path1, "wb") as f:
            f.write(uploaded_file1.read())
        with open(path2, "wb") as f:
            f.write(uploaded_file2.read())

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(path1), caption="Image 1", use_container_width=True)
        with col2:
            st.image(Image.open(path2), caption="Image 2", use_container_width=True)

        if st.button("Compare Faces"):
            desc1 = extract_landmark_hog_descriptors(path1)
            desc2 = extract_landmark_hog_descriptors(path2)

            if desc1 is None or desc2 is None:
                st.error("Could not process one or both images.")
            else:
                similarities = compare_hog_descriptors(desc1, desc2)
                avg_similarity = sum(similarities) / len(similarities)
                percentage = round(avg_similarity * 100, 2)

                st.success(f"Similarity score: {percentage}%")
                if percentage > 85:
                    st.info("Faces are very likely the same person.")
                elif percentage > 70:
                    st.warning("Faces might be the same person.")
                else:
                    st.error("Faces are likely different.")

                # графік
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(similarities, marker='o')
                ax.set_title("Cosine Similarity per Landmark")
                ax.set_xlabel("Landmark Index")
                ax.set_ylabel("Similarity")
                ax.set_ylim([0, 1])
                ax.grid(True)
                st.pyplot(fig)

# ========== HAAR + DNN ==========
elif option == "Face Detection (Haar + DNN)":
    st.subheader("Select one or more images for detection")
    detection_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if detection_files and st.button("Run Detection"):
        with st.spinner("Running detection..."):
            tmp_paths = []
            for file in detection_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(file.read())
                tmp_paths.append(tmp.name)

            results = run_face_detection(tmp_paths)

        for res in results:
            st.subheader(res['filename'])
            col1, col2 = st.columns(2)
            with col1:
                st.image(res['haar_img'], caption="Haar Result", use_container_width=True)
            with col2:
                st.image(res['dnn_img'], caption="DNN Result", use_container_width=True)

        fig1, fig2 = plot_detection_stats(results)
        st.pyplot(fig1)
        st.pyplot(fig2)

# ========== LANDMARKS ==========
elif option == "Face Landmarks (Dlib)":
    st.subheader("Select one or more images for landmark detection")
    landmark_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if landmark_files and st.button("Run Landmarks"):
        with st.spinner("Running landmark detection..."):
            tmp_paths = []
            for file in landmark_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(file.read())
                tmp_paths.append(tmp.name)

            results = run_landmarks(tmp_paths)

        for res in results:
            st.subheader(res["filename"])
            st.image(res["output_img"], caption="Landmarks", use_container_width=True)


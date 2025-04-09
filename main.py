def show_menu():
    print("Face Processing Utility")
    print("1. Face detection (Haar + DNN)")
    print("2. Face landmarks detection (Dlib)")
    print("3. Face comparison (embeddings)")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            try:
                from detectors.compare_detectors import run_face_detection
                run_face_detection()
            except ImportError:
                print("Module for face detection not found.")
        elif choice == "2":
            try:
                from detectors.landmarks import run_landmarks
                run_landmarks()
            except ImportError:
                print("Module for face landmarks not found.")
        elif choice == "3":
            try:
                from detectors.embeddings import run_embedding_compare
                run_embedding_compare()
            except ImportError:
                print("Module for face comparison not found.")
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("❗ Invalid option. Please try again.")

if __name__ == "__main__":
    main()

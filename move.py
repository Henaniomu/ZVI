import os
import shutil

base_dir = "images/lfw_funneled"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            source_path = os.path.join(root, file)
            target_path = os.path.join(base_dir, file)

            if source_path != target_path:
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} → {target_path}")

# for root, dirs, files in os.walk(base_dir, topdown=False):
#     if root != base_dir and not os.listdir(root):
#         os.rmdir(root)
#         print(f"Removed empty folder: {root}")

print("All photos successfully moved.")

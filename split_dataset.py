import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === 1️⃣ Set your base paths ===
base_dir = "/Users/arnabii/Documents/project/model/skin cancer detection dataset"
images_dir_1 = os.path.join(base_dir, "HAM10000_images_part_1")
images_dir_2 = os.path.join(base_dir, "HAM10000_images_part_2")
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")

# === 2️⃣ Output structure ===
output_dir = os.path.join(base_dir, "dataset_ready")
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

# === 3️⃣ Read metadata ===
df = pd.read_csv(metadata_path)
print(f"✅ Metadata loaded: {df.shape[0]} records")
print(df.head())

# === 4️⃣ Create folders for each class ===
classes = df['dx'].unique()
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# === 5️⃣ Helper: find image in either part_1 or part_2 ===
def find_image(image_id):
    file_name = f"{image_id}.jpg"
    path1 = os.path.join(images_dir_1, file_name)
    path2 = os.path.join(images_dir_2, file_name)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

# === 6️⃣ Split by class into train and val (80/20) ===
for cls in classes:
    cls_df = df[df['dx'] == cls]
    train_files, val_files = train_test_split(cls_df['image_id'], test_size=0.2, random_state=42)

    print(f"📂 Processing class: {cls} ({len(train_files)} train, {len(val_files)} val)")

    # Copy training images
    for img_id in train_files:
        src = find_image(img_id)
        if src:
            dst = os.path.join(train_dir, cls, f"{img_id}.jpg")
            shutil.copy(src, dst)

    # Copy validation images
    for img_id in val_files:
        src = find_image(img_id)
        if src:
            dst = os.path.join(val_dir, cls, f"{img_id}.jpg")
            shutil.copy(src, dst)

print("\n✅ DONE! Your dataset has been organized successfully!")
print(f"📁 Location: {output_dir}")
print("""
Structure:
dataset_ready/
 ├── train/
 │    ├── mel/
 │    ├── nv/
 │    ├── bkl/
 │    ├── akiec/
 │    ├── bcc/
 │    ├── df/
 │    └── vasc/
 └── val/
      ├── mel/
      ├── nv/
      ├── bkl/
      ├── akiec/
      ├── bcc/
      ├── df/
      └── vasc/
""")

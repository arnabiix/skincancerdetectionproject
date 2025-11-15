import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
model = timm.create_model('xception', pretrained=True)

# ---- Paths ----
BASE_DIR = "model/skin cancer detection dataset"
IMG_DIR_1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

# ---- Custom Dataset ----
class SkinDataset(Dataset):
    def __init__(self, df, img_dirs, transform=None):
        self.df = df
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"] + ".jpg"
        for dir_ in self.img_dirs:
            img_path = os.path.join(dir_, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                break
        label = self.df.iloc[idx]["dx"]
        label_idx = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label_idx


# ---- Read CSV ----
df = pd.read_csv(CSV_PATH)
label_map = {label: idx for idx, label in enumerate(df["dx"].unique())}
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["dx"], random_state=42)

# ---- Transforms ----
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---- Data Loaders ----
train_dataset = SkinDataset(df_train, [IMG_DIR_1, IMG_DIR_2], transform=train_transform)
val_dataset = SkinDataset(df_val, [IMG_DIR_1, IMG_DIR_2], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---- Model ----
device = torch.device("cpu")  # CPU-only for MacBook Intel

# Create Xception model using timm
model = timm.create_model('xception', pretrained=True, num_classes=len(label_map))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---- Training ----
EPOCHS = 3  # Try 3 epochs first to test pipeline

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ---- Save Model ----
torch.save(model.state_dict(), "xception_model.pth")
print("✅ Model saved as xception_model.pth")

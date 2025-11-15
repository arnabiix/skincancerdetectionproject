import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# === Paths ===
BASE_DIR = "model/skin cancer detection dataset"
IMG_DIR_1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

# === Custom Dataset ===
class SkinDataset(Dataset):
    def __init__(self, df, img_dirs, transform=None):
        self.df = df
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"] + ".jpg"
        label = self.df.iloc[idx]["dx"]
        label_idx = label_map[label]

        # Find the image in either directory
        for d in self.img_dirs:
            img_path = os.path.join(d, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                break

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# === Read CSV & Split ===
df = pd.read_csv(CSV_PATH)
label_map = {label: idx for idx, label in enumerate(df["dx"].unique())}
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["dx"], random_state=42)

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Datasets & Dataloaders ===
train_dataset = SkinDataset(df_train, [IMG_DIR_1, IMG_DIR_2], transform=train_transform)
val_dataset = SkinDataset(df_val, [IMG_DIR_1, IMG_DIR_2], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Model Setup (ResNet50) ===
device = torch.device("cpu")  # change to 'cuda' if you use GPU
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_map))  # replace final layer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# === Training Loop ===
EPOCHS = 3
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

    avg_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

    # === Validation ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"📊 Validation Accuracy: {val_acc:.2f}%")

# === Save Model ===
torch.save(model.state_dict(), "resnet_model.pth")
print("\n🎉 Training complete! Model saved as resnet_model.pth")

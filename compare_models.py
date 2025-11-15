import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from timm import create_model

# === Paths ===
BASE_DIR = "model/skin cancer detection dataset"
IMG_DIR_1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

# === Load CSV and split ===
df = pd.read_csv(CSV_PATH)
label_map = {label: idx for idx, label in enumerate(df["dx"].unique())}
reverse_map = {v: k for k, v in label_map.items()}

from sklearn.model_selection import train_test_split
_, df_val = train_test_split(df, test_size=0.2, stratify=df["dx"], random_state=42)

# === Dataset ===
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

        for d in self.img_dirs:
            img_path = os.path.join(d, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                break

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_dataset = SkinDataset(df_val, [IMG_DIR_1, IMG_DIR_2], transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cpu")

# === Helper function to evaluate model ===
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# === Load ResNet model ===
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, len(label_map))
resnet.load_state_dict(torch.load("resnet_model.pth", map_location=device))
resnet = resnet.to(device)

# === Load Xception model ===
xception = create_model("xception", pretrained=False, num_classes=len(label_map))
xception.load_state_dict(torch.load("xception_model.pth", map_location=device))
xception = xception.to(device)

# === Evaluate both ===
print("\n🔍 Evaluating ResNet model...")
resnet_acc = evaluate_model(resnet, val_loader)

print("\n🔍 Evaluating Xception model...")
xception_acc = evaluate_model(xception, val_loader)

# === Results ===
print("\n📊 Model Comparison Results:")
print(f"✅ ResNet50 Accuracy: {resnet_acc:.2f}%")
print(f"✅ Xception Accuracy: {xception_acc:.2f}%")

if resnet_acc > xception_acc:
    print("\n🏆 ResNet50 performed better!")
else:
    print("\n🏆 Xception performed better!")


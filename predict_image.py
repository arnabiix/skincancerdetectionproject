import torch
from timm import create_model
from PIL import Image
from torchvision import transforms
import pandas as pd

# 1️⃣ Load trained model
model = create_model('xception', pretrained=False, num_classes=7)
model.load_state_dict(torch.load('xception_model.pth', map_location='cpu'))
model.eval()

print("✅ Model loaded successfully and ready for prediction!")

# 2️⃣ Load and preprocess the image
image_path = "model/skin cancer detection dataset/HAM10000_images_part_1/ISIC_0024315.jpg"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image_tensor = transform(image).unsqueeze(0)

# 3️⃣ Make prediction
with torch.no_grad():
    outputs = model(image_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()

print(f"🩺 Predicted class index: {predicted_class}")

# 4️⃣ Map index to disease name
label_map = {
    0: 'bkl',
    1: 'nv',
    2: 'df',
    3: 'mel',
    4: 'vasc',
    5: 'bcc',
    6: 'akiec'
}

full_names = {
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi (moles)",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / Intraepithelial carcinoma"
}

predicted_label = label_map[predicted_class]
print(f"🧾 Predicted disease code: {predicted_label}")
print(f"💊 Full diagnosis: {full_names[predicted_label]}")

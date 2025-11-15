import os
from flask import Flask, render_template, request, redirect, url_for
from timm import create_model
import torch
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model_path = "xception_model.pth"
model = create_model('xception', pretrained=False, num_classes=7)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Label mapping (same as your training)
label_map = {0: 'bkl', 1: 'nv', 2: 'df', 3: 'mel', 4: 'vasc', 5: 'bcc', 6: 'akiec'}

# Image transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess the image
        image = Image.open(filepath).convert('RGB')
        img_t = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_t)
            _, predicted = torch.max(outputs, 1)
            label = label_map[predicted.item()]

        return render_template('index.html', result=label, image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)

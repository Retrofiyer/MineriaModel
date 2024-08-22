from flask import Flask, request, jsonify
import torch
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
import torch.nn as nn

app = Flask(__name__)

def predict_image(model, image, device):
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_probability = max(probabilities.squeeze().tolist())
        predicted_index = probabilities.squeeze().tolist().index(max_probability)

    probability_percentage = int(max_probability * 100)
    return predicted_index, probability_percentage

def load_model_for_inference(model_path, num_classes, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the model
model_path = './modelos/classification_model4_4.pth'
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model_for_inference(model_path, num_classes, device)

class_names = ['Cocker', 'Pekinese', 'Poodle', 'Schnauzer']

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image = Image.open(BytesIO(request.files['file'].read()))
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 400

    predicted_index, probability_percentage = predict_image(model, image, device)

    if probability_percentage < 80.99:
        return jsonify({"breed": "Undefined or invalid image", "probability": None})

    breed_description = class_names[predicted_index]
    
    return jsonify({"breed": breed_description, "probability": probability_percentage})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
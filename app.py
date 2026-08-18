from flask import (
    Flask,
    render_template,
    request
)

import torch

from torchvision import transforms

from PIL import Image

from model import CatDogCNN

app = Flask(__name__)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)
classes = [
    "cat",
    "dog",
    
]
model = CatDogCNN(
    num_classes=len(classes)
)
model.load_state_dict(
    torch.load(
        "models/best_cat_dog_model.pth",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()
print("Model loaded successfully.")
transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])
@app.route("/")
def home():
    return render_template(
        "index.html"
    )
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
     #get file

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]
    if file.filename == "":
        return "No image selected."
    image = Image.open(
        file
    ).convert("RGB")
    image_tensor = transform(
        image
    )
    image_tensor = image_tensor.unsqueeze(
        0
    )
    image_tensor = image_tensor.to(
        device
    )
    #prediction
    with torch.no_grad():
        logits = model(
            image_tensor
        )
        probabilities = torch.softmax(
            logits,
            dim=1
        )
        prediction = torch.argmax(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()
    predicted_class = classes[
        predicted_index
    ]
    confidence = probabilities[
        0,
        predicted_index
    ].item()
    confidence = confidence * 100
#return result
    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=f"{confidence:.2f}%"
    )
# run
if __name__ == "__main__":

    app.run(
        debug=True
    )
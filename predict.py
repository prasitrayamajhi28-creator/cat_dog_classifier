import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classes = [
    
    "cat",
    "dog",
    
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = CatDogCNN(
    num_classes=2
)
model.load_state_dict(
    torch.load(
        "best_cat_dog_model.pth",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()
image = Image.open("IMG_9103.jpeg").convert("RGB")

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
print("Input shape:", image_tensor.shape)
image_tensor = image_tensor.to(device)
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(
        logits,
        dim=1
    )
    prediction = torch.argmax(
        probabilities,
        dim=1
    )
predicted_class = classes[
    prediction.item()
]
confidence = probabilities[
    0,
    prediction.item()
].item()

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence * 100:.2f}%")

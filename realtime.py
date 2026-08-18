import cv2
import torch
from torchvision import transforms
from model import CatDogCNN
#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

classes = [
   
    "cat",
    "dog"
]

transform = transforms.Compose([
    transforms.ToPILImage(),
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
#load best model
model.load_state_dict(
    torch.load(
        "models/best_cat_dog_model.pth",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()
print("Model loaded.")

#open webcamp
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam started.")
print("Press Q to quit.")

#Real time loop
while True:
    #capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    #Flip horizontally
    frame = cv2.flip(
        frame,
        1
    )
#conver brg to rgb
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    #transform image
    image_tensor = transform(
        rgb_frame
    )
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
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
    predicted_class = classes[
        prediction.item()
    ]
    confidence = probabilities[
        0,
        prediction.item()
    ].item()

  #display text
    text = (
        f"{predicted_class} "
        f"{confidence * 100:.1f}%"
    )
    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )
   #display frame
    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#release webcamp
cap.release()
cv2.destroyAllWindows()
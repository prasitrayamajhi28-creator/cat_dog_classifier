import cv2
import torch
from torchvision import transforms
from model import CatDogCNN

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
    num_classes=3
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
print("Model loaded.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam started.")
print("Put picture inside the box.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(
        frame,
        1
    )
    height, width, _ = frame.shape
   #define roi
    roi_width = 400
    roi_height = 400
    x1 = (
        width - roi_width
    ) // 2
    y1 = (
        height - roi_height
    ) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )
    roi = frame[
        y1:y2,
        x1:x2
    ]
    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )
    image_tensor = transform(
        roi_rgb
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
    cv2.putText(
        frame,
        "Place picture inside box | Q = quit",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.imshow(
        "Cat Dog recognization",
        frame
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
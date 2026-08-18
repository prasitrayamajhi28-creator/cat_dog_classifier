import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from dataset import test_loader
from model import CatDogCNN

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classes = [
    
   "dog",
   "cat"
    
]

model = CatDogCNN(
    num_classes=2
)

#Load best model
model.load_state_dict(
    torch.load(
        "models/best_cat_dog_model.pth",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()
print("Best model loaded.")
#store prediction
all_predictions = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        predictions = torch.argmax(
            logits,
            dim=1
        )
        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            y_batch.cpu().numpy()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)
print("\nTest Accuracy:")
print( f"{accuracy * 100:.2f}%")

#Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)
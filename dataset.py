import torch

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

dataset = datasets.ImageFolder(
    "data"
)
from collections import Counter

counts = Counter(dataset.targets)

print("Total images:", len(dataset))

for class_name, class_index in dataset.class_to_idx.items():
    print(f"{class_name}: {counts[class_index]}")
train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Load dataset



#inforamation

print("Total images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

#split
total_size = len(dataset)

train_size = int(0.70 * total_size)

val_size = int(0.15 * total_size)

test_size = (
    total_size
    - train_size
    - val_size
)
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_dataset.dataset=datasets.ImageFolder(
   "data",
    transform=train_transform
)
val_dataset.dataset = datasets.ImageFolder(
    "data",
    transform=val_test_transform
)
test_dataset.dataset = datasets.ImageFolder(
    "data",
    transform=val_test_transform
)
#dataloaders

batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
#final info

print("\nDataset split:")

print("Training:", len(train_dataset))

print("Validation:", len(val_dataset))

print("Testing:", len(test_dataset))
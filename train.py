import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mood_resnet18.pth"
CLASS_MAP = MODEL_DIR / "class_map.json"

IMG_SIZE = 224
BATCH = 32
EPOCHS = 15
LR = 1e-4
PATIENCE = 5   # Early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- DATASET ----------------

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

classes = train_dataset.classes
print("Classes:", classes)

with open(CLASS_MAP, "w") as f:
    json.dump({"classes": classes}, f)

# ---------------- CLASS WEIGHTS ----------------

labels = train_dataset.targets

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# ---------------- MODEL ----------------

model = models.resnet18(pretrained=True)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 layers (important for better learning)
for param in model.layer4.parameters():
    param.requires_grad = True

# Modify classifier
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes))
)

model = model.to(device)

# Train only unfrozen layers
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# Scheduler (VERY IMPORTANT)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.3
)

# ---------------- TRAINING ----------------

best_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    train_correct = 0

    for imgs, labels in train_loader:

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        train_loss += loss.item() * imgs.size(0)
        train_correct += torch.sum(preds == labels)

    train_loss /= len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            val_loss += loss.item() * imgs.size(0)
            val_correct += torch.sum(preds == labels)

    val_loss /= len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)

    print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    # Scheduler step
    scheduler.step(val_loss)

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": classes
        }, MODEL_PATH)

        print("✅ Best model saved")

    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= PATIENCE:
        print("⛔ Early stopping triggered")
        break

print("\n🎯 Training Complete")
print("Best Validation Accuracy:", best_acc)

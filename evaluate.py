import torch
import json
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------- PATHS --------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "mood_resnet18.pth"
CLASS_MAP = ROOT / "models" / "class_map.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD CLASSES --------
with open(CLASS_MAP) as f:
    class_names = json.load(f)["classes"]

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------- DATA --------
dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# -------- MODEL --------
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(class_names))
)

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

model.to(device)
model.eval()

# -------- EVALUATION --------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:

        imgs = imgs.to(device)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# -------- REPORT --------
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

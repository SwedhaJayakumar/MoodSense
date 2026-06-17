import cv2
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from collections import deque

# -------- PATHS --------
ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "mood_resnet18.pth"
CLASS_MAP = ROOT / "models" / "class_map.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD CLASSES --------
with open(CLASS_MAP) as f:
    class_names = json.load(f)["classes"]

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

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------- FACE DETECTOR --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- SMOOTHING BUFFER --------
buffer = deque(maxlen=5)

# -------- WEBCAM --------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        inp = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(inp)
            probs = torch.softmax(output, dim=1)[0]

            idx = torch.argmax(probs).item()
            conf = float(probs[idx])

        buffer.append((idx, conf))

        # -------- SMOOTH PREDICTION --------
        counts = {}
        for i, c in buffer:
            counts[i] = counts.get(i, 0) + 1

        best_idx = max(counts, key=counts.get)

        avg_conf = np.mean([c for i, c in buffer if i == best_idx])

        mood = class_names[best_idx]

        text = f"{mood} ({avg_conf*100:.1f}%)"

        # -------- DRAW --------
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

    cv2.imshow("Mood Detection (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

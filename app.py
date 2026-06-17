from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, re, json, datetime, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path

# -------- PATHS --------
ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "mood_resnet18.pth"
CLASS_MAP = ROOT / "models" / "class_map.json"

# -------- FLASK --------
app = Flask(__name__)
CORS(app)

# -------- LOAD EMOTION MODEL (UNCHANGED) --------
with open(CLASS_MAP, "r") as f:
    class_names = json.load(f)["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# -------- AI MODEL (FIXED) --------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
ai_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -------- AI FUNCTION --------
def get_ai_recommendation(mood, time):
    
    try:
        # -------- BIGGER DYNAMIC DATA --------
        food_map = {
            "happy": ["Ice cream", "Fruit salad", "Pizza", "Burger", "Milkshake"],
            "sad": ["Chocolate", "Warm soup", "Tea", "Coffee", "Comfort food"],
            "angry": ["Juice", "Fruits", "Salad", "Cold drinks"],
            "neutral": ["Rice with curry", "Balanced meal", "Chapati"],
            "fear": ["Herbal tea", "Dry fruits", "Warm milk"],
            "surprise": ["Snacks", "Sweets", "Biscuits"]
        }

        activity_map = {
            "happy": ["Go out with friends", "Watch a movie", "Play games"],
            "sad": ["Listen to music", "Take rest", "Watch videos"],
            "angry": ["Deep breathing", "Go for a walk", "Exercise"],
            "neutral": ["Read a book", "Study", "Do light work"],
            "fear": ["Meditation", "Talk to someone", "Relax"],
            "surprise": ["Take a walk", "Explore something new"]
        }

        tip_map = {
            "happy": ["Enjoy the moment", "Spread positivity"],
            "sad": ["Things will get better", "Stay strong"],
            "angry": ["Stay calm and breathe", "Control your thoughts"],
            "neutral": ["Stay balanced", "Keep going"],
            "fear": ["Be brave", "You are safe"],
            "surprise": ["Stay present", "Take it easy"]
        }

        # -------- TIME BASED CHANGE --------
        if time == "morning":
            extra_food = ["Idli", "Dosa", "Upma"]
        elif time == "afternoon":
            extra_food = ["Meals", "Rice", "Curd rice"]
        elif time == "evening":
            extra_food = ["Snacks", "Tea", "Coffee"]
        else:
            extra_food = ["Light dinner", "Milk", "Fruits"]

        # -------- RANDOM SELECTION --------
        food = random.choice(food_map.get(mood, ["Healthy food"]) + extra_food)
        activity = random.choice(activity_map.get(mood, ["Relax"]))
        tip = random.choice(tip_map.get(mood, ["Stay positive"]))

        # -------- AI ENHANCEMENT (SAFE) --------
        prompt = f"Give one short motivational sentence for a {mood} person."

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = ai_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8
        )

        ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nAI TIP:", ai_text)

        # Replace tip only if good
        if len(ai_text.split()) > 4:
            tip = ai_text.strip()
            
        print("FINAL OUTPUT:", food, activity, tip)

        return food, activity, tip

    except Exception as e:
        print("AI ERROR:", e)

        return (
            "Fresh juice",
            "Take a short walk",
            "Stay calm and positive"
        )

# -------- HELPERS --------
def decode_image(data_url):
    payload = re.sub(r"^data:image/.+;base64,", "", data_url)
    img_bytes = base64.b64decode(payload)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        return img

    x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
    return img[y:y+h, x:x+w]

def get_time_bucket(hour):
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 15:
        return "afternoon"
    elif 15 <= hour < 18:
        return "evening"
    else:
        return "night"

# -------- API --------
@app.route("/api/detect", methods=["POST"])
def detect():

    data = request.get_json()
    img_data = data.get("image")

    if not img_data:
        return jsonify({"ok": False, "error": "No image"}), 400

    img = decode_image(img_data)
    face = detect_face(img)

    # -------- PREDICTION --------
    inp = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)
        probs = torch.softmax(output, dim=1)[0]
        idx = torch.argmax(probs).item()

    mood = class_names[idx]
    conf = float(probs[idx])
    
    # -------- TIME --------
    hour = datetime.datetime.now().hour
    time_bucket = get_time_bucket(hour)

    # -------- AI RECOMMENDATION --------
    food, activity, tip = get_ai_recommendation(mood, time_bucket)

    # -------- RESPONSE --------
    return jsonify({
    "ok": True,
    "mood": mood,
    "confidence": conf,
    "food": food,          # IMPORTANT (not foods)
    "activity": activity,
    "tip": tip,
    "time": time_bucket
})
    
from transformers import pipeline
import ollama
import wikipedia

emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=False
)

chat_history = []
emotion_history = []


def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"], result["score"]


def get_wiki_summary(topic):
    try:
        return wikipedia.summary(topic, sentences=2)
    except:
        return "Sorry, I couldn't find information."


@app.route("/api/chatbot", methods=["POST"])
def chatbot():

    data = request.json
    user_input = data["message"]

    emotion, confidence = detect_emotion(user_input)

    emotion_history.append(emotion)

    if "who is" in user_input.lower() or "tell me about" in user_input.lower():

        topic = user_input.lower().replace("who is","").replace("tell me about","")
        wiki = get_wiki_summary(topic)

        return jsonify({
            "reply": wiki,
            "emotion": emotion
        })


    if emotion == "sadness":
        prompt = f"User seems sad. Respond empathetically: {user_input}"

    elif emotion == "joy":
        prompt = f"User seems happy. Respond cheerfully: {user_input}"

    elif emotion == "anger":
        prompt = f"User seems angry. Respond calmly: {user_input}"

    elif emotion == "fear":
        prompt = f"User seems fearful. Respond reassuringly: {user_input}"

    else:
        prompt = user_input


    chat_history.append({"role":"user","content":prompt})

    response = ollama.chat(
        model="llama3",
        messages=chat_history
    )

    reply = response["message"]["content"]

    chat_history.append({"role":"assistant","content":reply})

    return jsonify({
        "reply": reply,
        "emotion": emotion
    })

# -------- RUN --------
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)

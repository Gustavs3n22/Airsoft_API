from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import requests
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

# SQLite database setup
def init_db():
    conn = sqlite3.connect('api_logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT,
            payload TEXT,
            response TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

# Log to SQLite
def log_to_db(api_key, payload, response):
    try:
        conn = sqlite3.connect('api_logs.db')
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False)
        response_json = json.dumps(response, ensure_ascii=False)
        cursor.execute('''
            INSERT INTO api_logs (api_key, payload, response, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (api_key, payload_json, response_json, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()

valid_categories = ['Снаряжение и защита', 'Аксессуары и Запчасти', 'Страйкбольное оружие']

class CategoryClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = CategoryClassifier(model_name, num_labels=len(valid_categories))
model.load_state_dict(torch.load("pytorch_model.bin"))
model.to(device)
model.eval()

model_img = load_model('vgg19_model.keras')
with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

def predict_text_category(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    predictions = [
        {"категория": valid_categories[i], "уверенность": float(probs[i])}
        for i in range(len(probs)) if probs[i] > 0.4
    ]
    
    if not predictions:
        return [{"категория": "не определено", "уверенность": 0.0}]
    
    return predictions

with open('image_classes.json', 'r', encoding='utf-8') as f:
    category_mapping = json.load(f)

def load_api_keys():
    try:
        with open("keys.json", "r") as file:
            data = json.load(file)
            return data.get("api_keys", [])
    except FileNotFoundError:
        print("Error: keys.json not found")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON in keys.json")
        return []

def preprocess_image(img, mean, std):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = np.expand_dims(img, axis=0)
    return img

def predict_photo_category(photo_urls):
    mean = np.array([103.939, 116.779, 123.68])
    std = np.array([1, 1, 1])
    results = []

    if isinstance(photo_urls, str):
        photo_urls = [photo_urls]

    for url in photo_urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                preprocessed_img = preprocess_image(img, mean, std)
                predictions = model_img.predict(preprocessed_img)

                predicted_class_index = np.argmax(predictions, axis=1)
                predicted_confidence = np.max(predictions, axis=1)
                predicted_labels = label_encoder.inverse_transform(predicted_class_index)

                result = {
                    "prediction": [
                        {
                            "url": url,
                            "класс": category_mapping.get(predicted_labels[i], "не определено"),
                            "подкласс": predicted_labels[i],
                            "уверенность": float(predicted_confidence[i])
                        }
                        for i in range(len(predicted_labels))
                    ]
                }
                results.append(result)
            else:
                results.append({"url": url, "error": "Unable to fetch image"})
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return results

@app.route('/api/predict', methods=['POST'])
def predict():
    valid_api_keys = load_api_keys()
    
    api_key = request.headers.get('X-API-Key') or request.json.get('api_key') if request.is_json else None
    
    if not api_key or api_key not in valid_api_keys:
        return jsonify({"error": "API key invalid or missing"}), 401

    if not request.is_json:
        return jsonify({"error": "Request should be JSON"}), 400
    
    data = request.json
    text = data.get('text', '')
    photo_urls = data.get('photo_urls', [])

    text_result = predict_text_category(text) if text else []
    photo_result = predict_photo_category(photo_urls) if photo_urls else []

    response_data = {
        'text_prediction': text_result,
        'photo_prediction': photo_result
    }

    log_to_db(api_key, data, response_data)

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
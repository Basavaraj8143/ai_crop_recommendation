from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')

# 1️⃣ Initialize Flask app first
app = Flask(__name__)

# 2️⃣ Enable CORS after app is created
CORS(app)

# Load ML model and label encoder
model = joblib.load(os.path.join(models_dir, 'crop_model.pkl'))
le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))

@app.route('/')
def home():
    return "AI Crop Recommendation API is running!"

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        input_df = pd.DataFrame([{
            'N': data['n'],
            'P': data['p'],
            'K': data['k'],
            'temperature': data['temperature'],
            'humidity': data['humidity'],
            'ph': data['ph'],
            'rainfall': data['rainfall']
        }])

        pred_label = model.predict(input_df)[0]
        top3_probs = model.predict_proba(input_df)[0]
        top3_idx = top3_probs.argsort()[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)

        confidence = round(max(top3_probs)*100, 2)

        return jsonify({
            'recommended_crop': le.inverse_transform([pred_label])[0],
            'top_3_crops': list(top3_crops),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

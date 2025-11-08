# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model bundle
bundle = joblib.load("model/crop_recommender.pkl")
model = bundle['model']
label_encoders = bundle.get('label_encoders', {})
crop_data = bundle['crop_data'].copy()
npk_max_sum = bundle.get('npk_max_sum', 1.0)

# helper to encode safely (if unseen, add and transform)
def safe_encode(colname, value):
    le = label_encoders.get(colname)
    if le is None:
        return 0
    val = str(value).lower().strip()
    # try direct match (we stored classes maybe lowercase?)
    classes = list(le.classes_)
    if val in classes:
        return int(le.transform([val])[0])
    # unseen: append to classes (keeps shape for transform)
    classes.append(val)
    le.classes_ = np.array(classes)
    return int(le.transform([val])[0])

@app.route('/')
def index():
    return render_template('index.html')  # your form

@app.route('/recommend', methods=['POST'])
def recommend():
    # read user form
    district = request.form.get('district', '').strip().lower()
    taluq = request.form.get('taluq', '').strip().lower()
    soil = request.form.get('soil_type', '').strip().lower()
    season = request.form.get('season', '').strip().lower()
    # normalize season similar to train
    if season.startswith('k'): season_norm = 'kharif'
    elif season.startswith('r'): season_norm = 'rabi'
    elif season.startswith('a'): season_norm = 'annual'
    elif '/' in season: season_norm = 'both'
    else: season_norm = 'unknown'

    # user soil test values
    try:
        user_N = float(request.form.get('n') or 0)
        user_P = float(request.form.get('p') or 0)
        user_K = float(request.form.get('k') or 0)
        user_ph = float(request.form.get('ph') or np.nan)
    except:
        user_N = user_P = user_K = 0
        user_ph = np.nan

    # encode categorical features using saved encoders
    dist_enc = safe_encode('district', district)
    taluk_enc = safe_encode('taluq', taluq)
    season_enc = safe_encode('season', season_norm)
    # For soil we may not have an encoder; we will use soil_match later

    # Prepare temp_diff/rain_diff placeholders = 0 (since model expects these numeric features).
    # A better approach would be to fetch the region row and compute temp_diff/rain_diff accurately.
    # We'll try to fetch region row from region_cleaned to compute temp/rain diffs if available.
    region_df = pd.read_csv("data/cleaned/region_cleaned.csv")
    # look up matching region row for season
    region_row = region_df[
        (region_df['district'].astype(str).str.lower().str.strip() == district) &
        (region_df['taluq'].astype(str).str.lower().str.strip() == taluq)
    ]
    if not region_row.empty:
        region_row = region_row.iloc[0]
        # choose temp according to season
        if season_norm == 'rabi':
            region_temp = region_row.get('avg_temp_rabi', np.nan)
        elif season_norm == 'kharif':
            region_temp = region_row.get('avg_temp_kharif', np.nan)
        else:
            region_temp = np.nanmean([region_row.get('avg_temp_rabi', np.nan), region_row.get('avg_temp_kharif', np.nan)])
        rainfall_scaled = float(region_row.get('avg_rainfall', 0)) / 10.0
    else:
        region_temp = np.nan
        rainfall_scaled = np.nan

    # For each crop, compute features expected by model and get probability
    candidates = []
    for _, crop in crop_data.iterrows():
        # compute temp_diff and rain_diff relative to crop optimal
        temp_mean = crop.get('temp_mean', np.nan)
        humidity_mean = crop.get('humidity_mean', np.nan)

        temp_diff = abs(region_temp - temp_mean) if not pd.isna(region_temp) and not pd.isna(temp_mean) else 0.0
        rain_diff = abs(rainfall_scaled - humidity_mean) if not pd.isna(rainfall_scaled) and not pd.isna(humidity_mean) else 0.0

        # soil match between user's selected soil and crop main soil
        soil_crop = str(crop.get('main_soiltype','')).strip().lower()
        soil_match = 1 if soil_crop and (soil in soil_crop) else 0

        # crop nutrient/pH features
        crop_N = float(crop.get('N', 0))
        crop_P = float(crop.get('P', 0))
        crop_K = float(crop.get('K', 0))
        crop_ph = float(crop.get('ph_mean', np.nan))

        # Build feature vector matching training order:
        x_row = [
            dist_enc,
            taluk_enc,
            season_enc,
            temp_diff,
            rain_diff,
            soil_match,
            crop_N,
            crop_P,
            crop_K,
            crop_ph if not pd.isna(crop_ph) else 0.0
        ]
        x_df = pd.DataFrame([x_row], columns=['district','taluq','season','temp_diff','rain_diff','soil_match','crop_N','crop_P','crop_K','crop_ph'])
        prob = model.predict_proba(x_df)[0][1]  # probability of being suitable

        # compute NPK/ph similarity score to user input (lower is better)
        npk_diff_sum = abs(crop_N - user_N) + abs(crop_P - user_P) + abs(crop_K - user_K)
        npk_norm = npk_diff_sum / max(npk_max_sum, 1.0)  # normalized 0..1 (0 best)
        ph_diff = abs(crop_ph - user_ph) if not pd.isna(crop_ph) and not pd.isna(user_ph) else 0.0
        ph_norm = ph_diff / 14.0  # normalize by pH range

        # combine model prob and user npk/ph similarity
        alpha = 0.6  # weight for ML model probability
        beta = 0.4   # weight for user NPK/ph similarity (lower is better)
        final_score = alpha * prob + beta * (1 - (npk_norm + ph_norm)/2.0)  # higher is better

        candidates.append({
            'crop': crop['crop'],
            'prob': float(prob),
            'npk_norm': float(npk_norm),
            'ph_norm': float(ph_norm),
            'final_score': float(final_score),
            'yield_qtl_per_acre': crop.get('yield_qtl_per_acre'),
            'price_per_qtl_rs': crop.get('price_per_qtl_rs')
        })

    # sort by final_score descending
    recommended = sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:5]

    return jsonify({
        'recommended_crops': recommended
    })

if __name__ == '__main__':
    app.run(debug=True)

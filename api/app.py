from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'energy_model.pkl')
model = joblib.load(MODEL_PATH)

STATIONS      = ['Stamping', 'Body Welding', 'Paint Shop', 'Trim Assembly',
                  'Chassis Mount', 'Final Assembly', 'Quality Check', 'EV Battery Fit']
VEHICLE_MODELS = ['Nexon EV', 'Punch EV', 'Tiago EV', 'Curvv EV']
SHIFTS        = ['Morning', 'Afternoon', 'Night']

BASELINES = {
    'Stamping': 348, 'Body Welding': 506, 'Paint Shop': 796,
    'Trim Assembly': 252, 'Chassis Mount': 326, 'Final Assembly': 300,
    'Quality Check': 172, 'EV Battery Fit': 358,
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required = ['station', 'vehicle_model', 'shift', 'workers_on_shift',
                 'ambient_temp_c', 'production_time_min']
    for f in required:
        if f not in data:
            return jsonify({'error': f'Missing field: {f}'}), 400

    df = pd.DataFrame([{
        'station':             data['station'],
        'vehicle_model':       data['vehicle_model'],
        'shift':               data['shift'],
        'workers_on_shift':    int(data['workers_on_shift']),
        'ambient_temp_c':      float(data['ambient_temp_c']),
        'production_time_min': float(data['production_time_min']),
    }])

    predicted_kwh = round(float(model.predict(df)[0]), 2)
    baseline      = BASELINES.get(data['station'], 350)
    saving_pct    = round((baseline - predicted_kwh) / baseline * 100, 1)

    return jsonify({
        'predicted_energy_kwh': predicted_kwh,
        'baseline_kwh':         baseline,
        'saving_pct':           saving_pct,
        'station':              data['station'],
        'shift':                data['shift'],
    })

@app.route('/options', methods=['GET'])
def options():
    return jsonify({
        'stations':       STATIONS,
        'vehicle_models': VEHICLE_MODELS,
        'shifts':         SHIFTS,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'energy_model.pkl'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
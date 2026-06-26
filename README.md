# Energy Consumption Optimization Through Predicted Production Time

A full-stack machine learning application built for Tata Motors Passenger Vehicles Limited, Sanand (Ahmedabad). The system predicts energy consumption at each manufacturing station based on predicted production time and operational parameters, enabling proactive energy planning and optimization across the plant floor.

---

## Overview

Manufacturing plants consume significant energy across multiple stations with varying production cycles. This project addresses the challenge of energy inefficiency by building a predictive model that estimates energy consumption before a production run begins — allowing shift supervisors to identify over-baseline runs and take corrective action in advance.

The application covers eight stations at the Sanand plant and supports all four current Tata EV models manufactured there.

---

## Features

- Predicts energy consumption (kWh) for any station, shift, vehicle model, and production time
- Compares predicted consumption against station-specific baselines
- Generates actionable optimization suggestions based on prediction context
- Displays station-wise energy breakdown and hourly trend charts
- Logs all predictions in-session with CSV export
- REST API backend for integration with other plant systems

---

## Tech Stack

| Layer      | Technology                          |
|------------|-------------------------------------|
| ML Model   | scikit-learn (Gradient Boosting)    |
| Backend    | Python, Flask, Flask-CORS           |
| Frontend   | React, Recharts, Axios              |
| Data       | Pandas, NumPy (synthetic dataset)   |

---

## Project Structure

```
tata_energy_project/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator (2000 records)
│   └── sanand_energy_data.csv    # Generated training data
├── model/
│   ├── train_model.py            # Model training script
│   └── energy_model.pkl          # Trained model (generated locally)
├── api/
│   └── app.py                    # Flask REST API
├── frontend/
│   └── src/
│       ├── App.js                # React dashboard
│       └── App.css               # Styles
└── requirements.txt
```

---

## Stations Covered

Stamping, Body Welding, Paint Shop, Trim Assembly, Chassis Mount, Final Assembly, Quality Check, EV Battery Fit

## Vehicle Models

Nexon EV, Punch EV, Tiago EV, Curvv EV

---

## Model Performance

| Metric        | Value  |
|---------------|--------|
| Algorithm     | Gradient Boosting Regressor |
| R-squared     | 0.9770 |
| MAE           | ~12 kWh |
| Training size | 2000 records |
| CV folds      | 5      |

The model is trained on synthetic data modeled on real operational parameters (station type, vehicle model, shift, ambient temperature, worker count, production time). R-squared is expected to settle at 0.80-0.90 when retrained on real plant sensor data.

---

## Local Setup

### Prerequisites

- Python 3.9 or above
- Node.js 18 or above

### Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 data/generate_data.py
python3 model/train_model.py
python3 api/app.py
```

Flask will start at `http://localhost:5000`.

### Frontend

```bash
cd frontend
npm install
npm start
```

React will start at `http://localhost:3000`.

---

## API Reference

### POST /predict

Predicts energy consumption for a given production run.

**Request body:**
```json
{
  "station": "Paint Shop",
  "vehicle_model": "Nexon EV",
  "shift": "Morning",
  "workers_on_shift": 8,
  "ambient_temp_c": 36,
  "production_time_min": 75
}
```

**Response:**
```json
{
  "predicted_energy_kwh": 829.86,
  "baseline_kwh": 796,
  "saving_pct": -4.3,
  "station": "Paint Shop",
  "shift": "Morning"
}
```

### GET /options

Returns valid values for station, vehicle model, and shift.

### GET /health

Health check endpoint.

---

## Internship Context

Built as part of an internship at Tata Motors Passenger Vehicles Limited, Sanand, Ahmedabad. The project is greenfield and end-to-end, covering data generation, model training, API development, and frontend dashboard.

---

## Author

Vanshika Deswal

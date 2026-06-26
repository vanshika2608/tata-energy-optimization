import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 2000

stations = {
    'Stamping':      {'base_energy': 180, 'base_time': 35, 'energy_per_min': 4.8},
    'Body Welding':  {'base_energy': 220, 'base_time': 55, 'energy_per_min': 5.2},
    'Paint Shop':    {'base_energy': 310, 'base_time': 80, 'energy_per_min': 6.1},
    'Trim Assembly': {'base_energy': 95,  'base_time': 45, 'energy_per_min': 2.8},
    'Chassis Mount': {'base_energy': 130, 'base_time': 40, 'energy_per_min': 3.5},
    'Final Assembly':{'base_energy': 110, 'base_time': 50, 'energy_per_min': 2.9},
    'Quality Check': {'base_energy': 60,  'base_time': 25, 'energy_per_min': 2.2},
    'EV Battery Fit':{'base_energy': 145, 'base_time': 38, 'energy_per_min': 4.0},
}

shifts = ['Morning', 'Afternoon', 'Night']
shift_factor = {'Morning': 1.00, 'Afternoon': 1.06, 'Night': 0.88}

vehicle_models = ['Nexon EV', 'Punch EV', 'Tiago EV', 'Curvv EV']
model_factor   = {'Nexon EV': 1.10, 'Punch EV': 1.00, 'Tiago EV': 0.92, 'Curvv EV': 1.15}

rows = []
for _ in range(n):
    station  = np.random.choice(list(stations.keys()))
    s        = stations[station]
    shift    = np.random.choice(shifts)
    model    = np.random.choice(vehicle_models)
    workers  = np.random.randint(3, 12)
    temp     = round(np.random.uniform(28, 42), 1)   # Sanand summer range

    prod_time = max(10, s['base_time'] + np.random.normal(0, 8))

    # Realistic noise sources:
    # 1. Sensor measurement error (~2% of reading)
    sensor_noise = np.random.normal(0, s['base_energy'] * 0.02)
    # 2. Occasional machine inefficiency spikes (5% of runs)
    spike = np.random.choice([0, 1], p=[0.95, 0.05]) * np.random.uniform(30, 80)
    # 3. Worker efficiency variation (more workers = slightly less energy per min)
    worker_effect = -0.4 * (workers - 7)
    # 4. Unexplained plant-level variance (power grid fluctuations, etc.)
    plant_noise = np.random.normal(0, 25)

    energy = (
        s['base_energy']
        + s['energy_per_min'] * prod_time
        + shift_factor[shift] * 15
        + model_factor[model] * 20
        + 0.8 * temp
        + worker_effect
        + sensor_noise
        + spike
        + plant_noise
    )
    energy = max(30, round(energy, 2))

    rows.append({
        'station':              station,
        'vehicle_model':        model,
        'shift':                shift,
        'workers_on_shift':     workers,
        'ambient_temp_c':       temp,
        'production_time_min':  round(prod_time, 1),
        'energy_consumed_kwh':  energy,
    })

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
df.to_csv('data/sanand_energy_data.csv', index=False)
print(f"Generated {len(df)} records")
print(df.head())
print("\nStats:")
print(df[['production_time_min', 'energy_consumed_kwh', 'ambient_temp_c']].describe().round(2))
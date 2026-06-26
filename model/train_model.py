import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('data/sanand_energy_data.csv')

X = df.drop(columns=['energy_consumed_kwh'])
y = df['energy_consumed_kwh']

categorical_cols = ['station', 'vehicle_model', 'shift']
numerical_cols   = ['workers_on_shift', 'ambient_temp_c', 'production_time_min']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
cv   = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

print("=== Model Performance ===")
print(f"  MAE  : {mae:.2f} kWh")
print(f"  R²   : {r2:.4f}")
print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/energy_model.pkl')
print("\nModel saved to model/energy_model.pkl")

print("\n=== Sample Predictions ===")
sample = pd.DataFrame([{
    'station': 'Paint Shop',
    'vehicle_model': 'Nexon EV',
    'shift': 'Morning',
    'workers_on_shift': 8,
    'ambient_temp_c': 36.0,
    'production_time_min': 75.0,
}])
pred = pipeline.predict(sample)[0]
print(f"  Paint Shop | Nexon EV | Morning | 75 min → {pred:.1f} kWh")

sample2 = pd.DataFrame([{
    'station': 'Body Welding',
    'vehicle_model': 'Tiago EV',
    'shift': 'Night',
    'workers_on_shift': 6,
    'ambient_temp_c': 30.0,
    'production_time_min': 50.0,
}])
pred2 = pipeline.predict(sample2)[0]
print(f"  Body Welding | Tiago EV | Night | 50 min → {pred2:.1f} kWh")
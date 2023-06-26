




from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import uvicorn
from typing import List, Union
import os

app = FastAPI(title="Get Around API", version="1.0")

class CarFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool

def transform_input(car: CarFeatures):
    # define the fields for the dataframe
    fields = [
        'mileage', 'engine_power', 'private_parking_available', 'has_gps',
        'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
        'has_speed_regulator', 'fuel_electro', 'fuel_hybrid_petrol',
        'fuel_petrol', 'car_type_coupe', 'car_type_estate',
        'car_type_hatchback', 'car_type_sedan', 'car_type_subcompact',
        'car_type_suv', 'car_type_van', 'BMW', 'Citroën', 'Ferrari', 'Mercedes',
        'Mitsubishi', 'Nissan', 'Peugeot', 'Porsche', 'Renault', 'Toyota',
        'Volkswagen', 'other_cars'
    ]

    # initialize all fields to 0
    data = {field: 0 for field in fields}

    # set the fields based on the input
    data['mileage'] = car.mileage
    data['engine_power'] = car.engine_power
    data['private_parking_available'] = int(car.private_parking_available)
    data['has_gps'] = int(car.has_gps)
    data['has_air_conditioning'] = int(car.has_air_conditioning)
    data['automatic_car'] = int(car.automatic_car)
    data['has_getaround_connect'] = int(car.has_getaround_connect)
    data['has_speed_regulator'] = int(car.has_speed_regulator)
    
    # set the appropriate fuel field based on the fuel type
    if car.fuel in data:
        data[f"fuel_{car.fuel}"] = 1
    
    # set the appropriate car_type field based on the car type
    if f"car_type_{car.car_type}" in data:
        data[f"car_type_{car.car_type}"] = 1
    
    # set the appropriate brand field based on the brand name
    if car.model_key in data:
        data[car.model_key] = 1
    else:
        data['other_cars'] = 1

    return pd.DataFrame(data, index=[0])

@app.post("/predict")
async def predict(car: CarFeatures):
    # Load model and scaler
    xgb_model = joblib.load('xgboost_model.joblib')
    scaler = joblib.load('scaler.joblib')

    # Transform the input
    df_input = transform_input(car)

    # Scale the input data
    scaled_X_new = scaler.transform(df_input)

    # Make predictions with the loaded model
    y_pred = xgb_model.predict(scaled_X_new)

    # Format and return response
    response = {"prediction": y_pred.tolist()}

    return response

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))
   uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")


#curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"model_key\": \"BMW\", \"mileage\": 20000, \"engine_power\": 140, \"fuel\": \"petrol\", \"car_type\": \"sedan\", \"private_parking_available\": true, \"has_gps\": true, \"has_air_conditioning\": true, \"automatic_car\": true, \"has_getaround_connect\": true, \"has_speed_regulator\": true}"
#{"prediction":[163.30746459960938]}
#{
  #  "model_key": "BMW",
   # "mileage": 120000.0,
   # "engine_power": 140.0,
   # "fuel": "petrol",
   # "car_type": "sedan",
   # "private_parking_available": true,
   # "has_gps": true,
   # "has_air_conditioning": true,
   # "automatic_car": true,
   # "has_getaround_connect": true,
   # "has_speed_regulator": true
#}

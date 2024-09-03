from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import json
import numpy as np

app = FastAPI()

# Cargar el modelo TensorFlow
model = tf.keras.models.load_model("model.h5")

# Cargar scaler y label encoder
with open('scaler.json') as f:
    scaler = json.load(f)

with open('label_encoder.json') as f:
    label_encoder = json.load(f)

class PredictionRequest(BaseModel):
    hum: float
    luz: float
    pres: float
    temp: float
    vel: float

@app.get("/predict")
async def predict(hum: float, luz: float, pres: float, temp: float, vel: float):
    try:
        # Validación de parámetros
        if any(map(lambda x: not isinstance(x, (int, float)), [hum, luz, pres, temp, vel])):
            raise HTTPException(status_code=400, detail="Invalid input parameters. Ensure all parameters are numbers.")
        
        # Preprocesar la entrada
        input_data = [hum, luz, pres, temp, vel]
        scaled_input = [(val - scaler['mean'][i]) / scaler['scale'][i] for i, val in enumerate(input_data)]

        # Ejecutar predicción
        input_tensor = np.array([scaled_input], dtype=np.float32)
        prediction = model.predict(input_tensor)
        pred_class = np.argmax(prediction, axis=-1)[0]

        # Decodificar la predicción
        result = label_encoder['classes'][int(pred_class)]

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

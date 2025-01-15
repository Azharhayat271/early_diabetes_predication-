from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model_filename = 'random_forest_model.pkl'
loaded_model = joblib.load(model_filename)

# Create the FastAPI app
app = FastAPI()

# Define the input schema
class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str

# Encoding function
def encode_input(data):
    # Encoding mappings
    encoding_map = {'yes': 1, 'no': 0, 'male': 0, 'female': 1}

    # Convert yes/no strings to 1/0 and Gender to 0/1
    encoded = [
        data.Age,
        encoding_map[data.Gender.lower()],
        encoding_map[data.Polyuria.lower()],
        encoding_map[data.Polydipsia.lower()],
        encoding_map[data.sudden_weight_loss.lower()],
        encoding_map[data.weakness.lower()],
        encoding_map[data.Polyphagia.lower()],
        encoding_map[data.Genital_thrush.lower()],
        encoding_map[data.visual_blurring.lower()],
        encoding_map[data.Itching.lower()],
        encoding_map[data.Irritability.lower()],
        encoding_map[data.delayed_healing.lower()],
        encoding_map[data.partial_paresis.lower()],
        encoding_map[data.muscle_stiffness.lower()],
        encoding_map[data.Alopecia.lower()],
        encoding_map[data.Obesity.lower()]
    ]
    return np.array([encoded])

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Encode the input data
    try:
        encoded_input = encode_input(input_data)
    except KeyError as e:
        return {"error": f"Invalid input value: {e}"}

    # Predict using the loaded model
    prediction = loaded_model.predict(encoded_input)
    
    # Map prediction to class labels (e.g., Positive/Negative)
    class_map = {1: "Positive", 0: "Negative"}
    result = class_map[prediction[0]]
    
    return {"prediction": result}


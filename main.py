import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")

SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(features: IrisFeatures):
    X = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    prediction = model.predict(X)[0]
    return {"species": SPECIES[prediction]}

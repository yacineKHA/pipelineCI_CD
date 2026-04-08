import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None


SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(features: IrisFeatures):
    model = joblib.load("model.pkl")
    X = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    prediction = model.predict(X)[0]
    print(f"Predicted class index: {prediction}, Species: {SPECIES[prediction]}")
    return {"species": SPECIES[prediction]}

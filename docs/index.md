# Iris Classifier API

## Description

API de classification de fleurs Iris basée sur un modèle RandomForest entraîné avec scikit-learn.

## Endpoints

### `POST /predict`

Prédit l'espèce d'une fleur Iris à partir de ses mesures.

**Body :**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Réponse :**
```json
{
  "species": "Iris-setosa"
}
```

**Espèces possibles :** `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`

## Lancer en local

```bash
# Backend
uvicorn main:app --reload

# Frontend
streamlit run frontend.py
```

## Docker

```bash
docker build -f Dockerfile.backend -t iris-backend .
docker build -f Dockerfile.frontend -t iris-frontend .
```

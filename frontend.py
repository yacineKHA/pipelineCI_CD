import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("Iris Flower Classifier")
st.write("Renseigne de la fleur pour prédire l'espèce.")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Longueur sépale (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Largeur sépale (cm)", 2.0, 5.0, 3.5)

with col2:
    petal_length = st.slider("Longueur pétale (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Largeur pétale (cm)", 0.1, 2.5, 0.2)

if st.button("Prédire", type="primary"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        st.success(f"Espèce prédite : **{result['species']}**")
    except requests.exceptions.ConnectionError:
        st.error("Impossible de contacter le backend. Vérifie que FastAPI tourne sur le port 8000.")
    except Exception as e:
        st.error(f"Erreur : {e}")

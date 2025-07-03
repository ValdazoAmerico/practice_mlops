import json
import pandas as pd
from evaluate import load_model, evaluate_model
from deploy import deploy_model

import joblib

def load_model_joblib(path):
    return joblib.load(path)

# Cargar config
with open("thresholds.json") as f:
    config = json.load(f)

metric_name = config["metric"]
min_value = config["min_value"]

# Cargar data
test_data = pd.read_csv("data/test_sample.csv")

# Evaluar modelo candidato
candidate_model = load_model_joblib("models/candidato_model.pkl")
candidate_score = evaluate_model(candidate_model, test_data)

# Evaluar modelo actual
current_model = load_model("models/base_model.pkl")
current_score = evaluate_model(current_model, test_data)

print(f"🔍 Candidato: {candidate_score:.4f} | Actual: {current_score:.4f}")

# Validación
if candidate_score >= min_value and candidate_score >= current_score:
    deploy_model()
else:
    raise ValueError("❌ El modelo candidato no supera el umbral o es peor que el actual.")

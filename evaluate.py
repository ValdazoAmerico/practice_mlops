import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_model(path):
    return joblib.load(path)

def evaluate_model(model, test_data):
    print("model",model)
    # Convertir Species a números (0, 1, 2)
    label_encoder = LabelEncoder()
    #test_data["target"] = label_encoder.fit_transform(test_data["Species"])

    #print(test_data.head())
    #print(test_data.columns)
    print("len", len(test_data))

    # Separar features y target
    X = test_data.drop(["Species"], axis=1)
    y = test_data["Species"]

    print("X",X)

    # Hacer predicciones
    preds = model.predict(X)
    print("y_true",y)
    print("preds",preds)
    print("accuracy", accuracy_score(y, preds))

    return accuracy_score(y, preds)

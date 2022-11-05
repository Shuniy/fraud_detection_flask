from __future__ import unicode_literals
import json
from flask import Flask, request
import joblib
from settings import MODEL_FILENAME

app = Flask("Fraud Detection")

# load model at startup time
app.model = joblib.load(MODEL_FILENAME)

@app.route("/", methods = ["POST"])
def predict_fraud():
    input_data = request.get_json()
    if "features" not in input_data:
        return json.dumps({"error": "No features found in input"}), 400
    if not input_data["features"] or not isinstance(input_data["features"], list):
        return json.dumps({"error": "No feature values available"}), 400
    if isinstance(input_data["features"][0], list):
        results = app.model.predict_proba(input_data["features"]).tolist()
    else:
        results = app.model.predict_proba([input_data["features"]]).tolist()
    return json.dumps({"scores": [result[1] for result in results]}), 200

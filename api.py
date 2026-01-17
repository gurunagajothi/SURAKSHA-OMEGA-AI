from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("sos_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])
    return jsonify({"sos_level": pred})

app.run()

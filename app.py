import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Form
@app.route("/index")
def index():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    features = [[
        float(request.form["gender"]),
        float(request.form["car"]),
        float(request.form["realty"]),
        float(request.form["children"]),
        float(request.form["income"]),
        float(request.form["income_type"]),
        float(request.form["education"]),
        float(request.form["family_status"]),
        float(request.form["housing"]),
        float(request.form["days_employed"]),
        float(request.form["work_phone"]),
        float(request.form["phone"]),
        float(request.form["email"]),
        float(request.form["family_members"]),
        float(request.form["dependency"]),
        float(request.form["age"]),
        float(request.form["years_employed"]),
        float(request.form["months_balance"])
    ]]

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]

    return render_template("index.html", result=prediction)

# Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

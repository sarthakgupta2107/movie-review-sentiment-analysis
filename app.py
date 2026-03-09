from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    # Convert text to vector
    data = vectorizer.transform([review])

    # Predict sentiment
    prediction = model.predict(data)[0]

    # Convert result
    if prediction == 1:
        result = "Positive Review 😊"
    else:
        result = "Negative Review 😡"

    return render_template("index.html", result=result)


# Run app for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

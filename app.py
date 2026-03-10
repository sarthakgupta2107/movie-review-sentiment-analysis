from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"].lower()

    # simple rule check
    positive_words = ["good", "amazing", "great", "awesome", "fantastic", "love"]
    negative_words = ["bad", "boring", "worst", "hate", "poor"]

    if any(word in review for word in positive_words):
        result = "Positive 😊"
    elif any(word in review for word in negative_words):
        result = "Negative 😡"
    else:
        # ML model fallback
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]
        result = str(prediction)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

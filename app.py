from flask import Flask, request, jsonify
import pickle
import numpy as np
import gdown
import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Google Drive File ID
file_id = "1pShhKHhn6ViQ7YBCAblFLZRYV3St04Mn"
url = f"https://drive.google.com/uc?id={file_id}"

# Model file path
model_path = "random_forest_heart_disease.pkl"

# Download model only if not already present
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f())

print("Model loaded successfully!")


# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Heart Disease Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features (ensure the input keys match your training data)
        features = np.array(data["features"]).reshape(1, -1)  # Convert input to 2D array
        
        # Make prediction
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})  # Return prediction as JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
    

# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/add', methods=['POST'])
# def add_numbers():
#     data = request.get_json()
#     num1 = data.get("num1")
#     num2 = data.get("num2")

#     if num1 is None or num2 is None:
#         return jsonify({"error": "Missing numbers"}), 400

#     try:
#         result = float(num1) + float(num2)
#     except ValueError:
#         return jsonify({"error": "Invalid number format"}), 400

#     return jsonify({"result": result})

# if __name__ == '__main__':
#     app.run(debug=True)

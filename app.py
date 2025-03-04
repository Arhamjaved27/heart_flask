import gdown
import os
import pickle
import numpy as np
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

# Auto-detect number of features
try:
    num_features = model.n_features_in_  # Get number of features
except AttributeError:
    num_features = None  # Fallback if the attribute is missing

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # JSON input from the request

    # Auto-fetch required feature values
    if num_features is not None:
        input_features = list(data.values())[:num_features]  # Select only required features
    else:
        input_features = list(data.values())  # Use all available data

    # Convert input to numpy array and reshape for prediction
    input_array = np.array(input_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    return jsonify({"prediction": int(prediction[0])})

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

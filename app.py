import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route (POST Request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form.get("age"))
        bmi = float(request.form.get("bmi"))
        smoker = 1 if request.form.get("smoker") == "yes" else 0
        region = int(request.form.get("region"))  # Assuming model takes numerical input for region
        
        # Prepare input for model
        input_data = np.array([[age, bmi, smoker, region]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return JSON response
        return jsonify({"estimated_insurance_cost": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

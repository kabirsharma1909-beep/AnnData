import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- 1. Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- 2. Load the Trained Model ---
print("Loading the machine learning model...")
try:
    model = joblib.load('crop_recommendation_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'crop_recommendation_model.joblib' not found.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# --- 3. Define the Prediction Logic ---
def make_prediction(data):
    if model is None:
        return "Model not loaded", "Error"

    try:
        feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
        df = pd.DataFrame([data], columns=feature_columns)
        crop_prediction = model.predict(df)[0]

        pump_command = "PUMP_OFF"
        soil_moisture = data.get('soil_moisture', 100)
        
        if soil_moisture < 40:
            pump_command = "PUMP_ON"
        
        print(f"Prediction successful. Recommended Crop: {crop_prediction}, Pump Command: {pump_command}")
        return crop_prediction, pump_command

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error", "Error"

# --- 4. Web Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

# --- 5. API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    print("\nReceived a request on /predict")
    data = request.get_json()

    if not data:
        print("Error: No data received.")
        return jsonify({"error": "No input data provided"}), 400

    print("Data received:", data)
    crop, pump_action = make_prediction(data)
    
    return jsonify({
        'recommended_crop': crop,
        'pump_command': pump_action
    })

# --- 6. Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
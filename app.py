from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Model and scaler loaded successfully.")

except Exception as e:
    model = None
    scaler = None
    print(f"❌ Failed to load model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json()
        print("Received data:", data)

        # Extract and convert inputs
        age = float(data['Age'])
        sex = int(data['Sex'])
        chest_pain = int(data['ChestPainType'])
        cholesterol = float(data['Cholesterol'])
        max_hr = float(data['MaxHR'])
        exercise_angina = int(data['ExerciseAngina'])

        # Prepare input array
        input_data = np.array([[age, sex, chest_pain, cholesterol, max_hr, exercise_angina]])
        print("Raw input:", input_data)

        # Scale input
        input_scaled = scaler.transform(input_data)
        print("Scaled input:", input_scaled)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        print("Prediction:", prediction, "Probability:", probability)

        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability, 4)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

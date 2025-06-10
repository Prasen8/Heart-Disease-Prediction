import pickle
import numpy as np

try:
    with open(r'C:\Users\Prasen\OneDrive\Desktop\HD Prediction\model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")

    # Test with dummy input (6 features)
    test_input = np.array([[55, 1, 2, 240, 150, 0]])
    prediction = model.predict(test_input)
    proba = model.predict_proba(test_input)

    print("Prediction:", prediction)
    print("Probability:", proba)

except Exception as e:
    print("❌ Model loading or prediction failed:")
    print(e)

from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = load('E:/sentinel-gateway/model_111/fraud_detection_model.joblib')

# Endpoint to predict fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        df = pd.DataFrame([data])  # Wrap it in a list so it's treated as a single row

        # Preprocess the data like we did for training
        encoded_data = pd.get_dummies(df, columns=['location', 'transaction_type'], drop_first=True)

        # Ensure columns match training data
        required_columns = ['transaction_id', 'amount', 'location_Houston', 'location_Los Angeles', 'location_New York', 'location_San Francisco',
                            'transaction_type_in-person', 'transaction_type_online', 'transaction_type_wire-transfer']
        for col in required_columns:
            if col not in encoded_data:
                encoded_data[col] = 0

        # Reorder columns
        encoded_data = encoded_data[required_columns]

        # Feature scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(encoded_data)

        # Make predictions
        predictions = model.predict(scaled_data)

        # Add predictions to the data
        df['fraud_prediction'] = predictions.tolist()

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

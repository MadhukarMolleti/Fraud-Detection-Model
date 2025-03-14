import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the saved model
model = load('E:/sentinel-gateway/model_111/fraud_detection_model.joblib')
print("Model loaded successfully!")

# Load new data for testing
test_data = pd.read_csv('E:/sentinel-gateway/model_111/new_transaction_data.csv')

# Preprocess the data like we did for training
encoded_data = pd.get_dummies(test_data, columns=['location', 'transaction_type'], drop_first=True)

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

# Make predictions on scaled data
predictions = model.predict(scaled_data)

# Save predictions with the original data
test_data['fraud_prediction'] = predictions
test_data.to_csv('E:/sentinel-gateway/model_111/prediction_results.csv', index=False)

print("Predictions saved to prediction_results.csv")
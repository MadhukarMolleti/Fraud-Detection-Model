# Fraud Detection Model

## Overview
The AI Fraud Detection Model is intended to detect potentially fraudulent financial transactions with high accuracy. It uses machine learning algorithms to examine transaction data and forecast whether a transaction is valid or fraudulent.

## Features
- **Fraud Detection:** Forecasts the probability of a transaction being fraudulent.
- **High Accuracy:** Attained 91% accuracy on small data and 61% on larger data.
- **Real-time Prediction:** API-based model deployment for real-time fraud detection.
- **Secure Model:** Saved and served from a separate folder for security and convenience.

## Technologies Used
- **AI Model:** Trained with machine learning algorithms with Python.
- **Backend:** Flask to deploy the model and create APIs.
- **Model Management:** Joblib to save and load the trained model.
- **HTTP Requests:** Requests library to send data to the model.

## Key Files
- `fraud_detection_model.py`: Script to train and save the fraud detection model.
- `fraud_detection_model.joblib`: Saved machine learning model.
- `deploy_fraud_model.py`: Flask app for exposing the model through an API.
- `test_fraud_detection_model.py`: Script for testing the deployed model.
- `new_transaction_data.csv`: Sample transaction data for testing.
- `prediction_results.csv`: Output predictions from the model.

## Running the Project
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-fraud-detection.git
cd ai-fraud-detection
```

### 2. Run the API Server
```bash
python -m waitress --host=0.0.0.0 --port=5000 model_111.deploy_fraud_model:app
```

### 3. Send Test Request
```bash
python model_111/send_test_request.py
```

Or using PowerShell:
```powershell
curl -Method POST -Uri http://localhost:5000/predict -Body '{"transaction_id":12345,"amount":1000,"transaction_type":"online","location":"New York"}' -ContentType "application/json"
```

### 4. Verify Prediction Results
Successful response:
```json
[
  {
    "transaction_id": 12345,
    "amount": 1000,
    "transaction_type": "online",
    "location": "New York",
    "fraud_prediction": 0
  }
]
```

A `fraud_prediction` of `0` means a valid transaction, and `1` means suspected fraud.

## Future Development
- Improve model performance with improved data and feature engineering.
- Utilize more sophisticated AI methods for anomaly detection.
- Integrate with the Sentinel Gateway for integrated fraud prevention.

## Contributors
- [Mollet Madhukar](https://github.com/MadhukarMolleti)

import requests

url = "http://127.0.0.1:5000/predict"

# Example transaction data
data = {
    "transaction_id": 12345,
    "amount": 1000,
    "location": "New York",
    "transaction_type": "online"
}

response = requests.post(url, json=data)

print("Response:", response.json())

import requests

API_URL = "http://127.0.0.1:8000/forecast"

def fetch_prediction(payload):

    response = requests.post(
        API_URL,
        json=payload
    )

    return response.json()

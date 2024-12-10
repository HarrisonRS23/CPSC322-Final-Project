import requests
import json

# Define the local Flask app URL
url = "http://127.0.0.1:5001/predict?"

# Example query parameters
params = {
    "height_cm": 185,
    "positions": "Goalkeeper",
    "skill_moves": 3,
    "crossing": 72,
}

# Construct the query URL
url += "&".join([f"{key}={value}" for key, value in params.items()])
print(f"Request URL: {url}")

# Make the GET request
response = requests.get(url)

# Process the response
if response.status_code == 200:
    json_obj = json.loads(response.text)
    print("Prediction:", json_obj.get("prediction"))
else:
    print(f"Error {response.status_code}: {response.text}")
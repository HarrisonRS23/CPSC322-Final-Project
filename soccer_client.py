import requests
import json

# Define the local API base URL
url = "http://127.0.0.1:5000/predict"

# Example input values
example_data = {
    "height_cm": "180",
    "positions": "Forward",
    "skill_moves": "4",
    "crossing": "85",
    "finishing": "90",
    "short_passing": "80",
    "dribbling": "88",
    "freekick_accuracy": "75",
    "long_passing": "70",
    "ball_control": "86",
    "acceleration": "85",
    "agility": "84",
    "shot_power": "89",
    "stamina": "78",
    "long_shots": "80",
    "interceptions": "40",
    "positioning": "92",
    "vision": "88",
    "marking": "30",
    "standing_tackle": "50"
}

# Validate input data
missing_keys = [key for key, value in example_data.items() if not value]
if missing_keys:
    print(f"Error: Missing or invalid values for the following keys: {missing_keys}")
    exit(1)

# Display the request URL for debugging
request_url = f"{url}?{'&'.join([f'{key}={value}' for key, value in example_data.items()])}"
print(f"Request URL: {request_url}")

# Send the GET request
try:
    response = requests.get(url, params=example_data)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse and display the response
    response_data = response.json()
    prediction = response_data.get("prediction", "No prediction available")
    print("Predicted Soccer Position:", prediction)

except requests.exceptions.RequestException as e:
    print(f"HTTP Request failed: {e}")
except json.JSONDecodeError:
    print("Error decoding the response JSON.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

import requests  # a library for HTTP requests
import json  # a library for parsing strings/JSON objects

# Base URL for the soccer position prediction app
url = "https://cpsc322-final-project-f3on.onrender.com/predict?"

# Add soccer-specific query parameters
query_params = {
    "height_cm": 180,          # Example height in cm
    "short_passing": 85,       # Example short passing skill level
    "vision": 78,              # Example vision skill level
    "crossing": 70             # Example crossing skill level
}

# Append query parameters to the URL
query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
full_url = url + query_string
print(f"Request URL: {full_url}")

# Make the GET request
response = requests.get(full_url)

# First check the status code
print("Status code:", response.status_code)

# Handle successful response
if response.status_code == 200:
    # Parse the JSON response
    json_obj = json.loads(response.text)
    print("Response type:", type(json_obj))
    print("Response JSON:", json_obj)

    # Print the predicted position
    prediction = json_obj.get("prediction", "No prediction available")
    print("Predicted Soccer Position:", prediction)
else:
    # Handle errors
    print("Error: Failed to get a response from the server.")
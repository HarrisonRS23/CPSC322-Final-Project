import requests
import json
import pickle

# Load the header attributes dynamically from the pickled tree
def load_tree_header():
    try:
        with open("soccer_tree.p", "rb") as infile:
            header, _ = pickle.load(infile)
        return header
    except Exception as e:
        print(f"Error loading tree header: {e}")
        return []

header = load_tree_header()

# Base URL for the Flask app
url = "https://cpsc322-final-project-f3on.onrender.com/predict?"

# Get input for each attribute dynamically based on the header
query_params = {attr: input(f"Enter value for {attr}: ") for attr in header}

# Construct the query string
query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
full_url = url + query_string
print(f"Request URL: {full_url}")

# Make the GET request
response = requests.get(full_url)

# Check the response status
print("Status code:", response.status_code)

if response.status_code == 200:
    # Parse and display the JSON response
    json_obj = json.loads(response.text)
    print("Response JSON:", json_obj)
    prediction = json_obj.get("prediction", "No prediction available")
    print("Predicted Soccer Position:", prediction)
else:
    print("Error: Failed to get a response from the server.")
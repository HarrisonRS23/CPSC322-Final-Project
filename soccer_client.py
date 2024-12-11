import requests
import json
import pickle

def load_tree_header():
    """
    Load the header from the pickled decision tree.
    """
    try:
        with open("soccer_tree.p", "rb") as infile:
            header, _ = pickle.load(infile)
        return header
    except Exception as e:
        print(f"Error loading tree header: {e}")
        return []

header = load_tree_header()

# Define the API base URL
url = "https://cpsc322-final-project-f3on.onrender.com/predict?"

# Collect user input for each attribute
query_params = {attr: input(f"Enter value for {attr}: ") for attr in header}

# Construct the full query URL
query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
full_url = url + query_string
print(f"Request URL: {full_url}")

# Send the GET request
response = requests.get(full_url)

# Parse and display the response
if response.status_code == 200:
    json_obj = json.loads(response.text)
    prediction = json_obj.get("prediction", "No prediction available")
    print("Predicted Soccer Position:", prediction)
else:
    print("Error: Failed to get a response from the server.")
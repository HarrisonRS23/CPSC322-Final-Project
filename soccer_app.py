import os
import pickle
from flask import Flask, render_template, request, jsonify
from mysklearn.myclassifiers import MyDecisionTreeClassifier

app = Flask(__name__)

# Load the pickled decision tree and header
def load_tree():
    try:
        with open("soccer_tree.p", "rb") as infile:
            header, tree = pickle.load(infile)
        return header, tree
    except FileNotFoundError:
        print("Error: The file 'soccer_tree.p' was not found. Ensure the tree is pickled and saved.")
        return None, None
    except Exception as e:
        print(f"Error loading tree: {e}")
        return None, None

# Load the tree and header
header, soccer_tree = load_tree()
if header is None or soccer_tree is None:
    print("Failed to load decision tree or header. Exiting...")
    exit(1)

print("Loaded Header:", header)

# Attribute mapping for descriptive names
attribute_mapping = {
    "height_cm": "att0",
    "positions": "att1",
    "skill_moves(1-5)": "att2",
    "crossing": "att3",
    "finishing": "att4",
    "short_passing": "att5",
    "dribbling": "att6",
    "freekick_accuracy": "att7",
    "long_passing": "att8",
    "ball_control": "att9",
    "acceleration": "att10",
    "agility": "att11",
    "shot_power": "att12",
    "stamina": "att13",
    "long_shots": "att14",
    "interceptions": "att15",
    "positioning": "att16",
    "vision": "att17",
    "marking": "att18",
    "standing_tackle": "att19",
}

# Ensure headers use descriptive names
descriptive_header = [key for key in attribute_mapping.keys()]
print("Mapped Header (Descriptive):", descriptive_header)

# Initialize the decision tree classifier
tree_classifier = MyDecisionTreeClassifier()
tree_classifier.tree = soccer_tree  # Set the loaded tree
tree_classifier.header = header  # Use the original header (att0, att1, ...)

@app.route("/", methods=["GET", "POST"])
def index_page():
    prediction = ""
    if request.method == "POST":
        try:
            # Collect user inputs from the form
            instance = [request.form.get(attr, "") for attr in descriptive_header]
            if not all(instance):
                raise ValueError("Missing or invalid input data.")
            prediction = predict_player_position(instance)
        except Exception as e:
            print(f"Error processing form input: {e}")
            prediction = "Invalid input"
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Collect query parameters based on descriptive header
        instance = [request.args.get(attr, "") for attr in descriptive_header]
        print("Instance received:", instance)

        if not all(instance):  # Ensure all attributes are present
            return jsonify({"error": "Missing or invalid input data"}), 400

        # Predict the position
        prediction = predict_player_position(instance)
        if prediction is not None:
            return jsonify({"prediction": prediction}), 200
        return jsonify({"error": "Prediction could not be made"}), 400
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def predict_player_position(instance):
    """
    Predict the player's position using the decision tree.
    """
    try:
        # Convert descriptive attributes to generic names for prediction
        mapped_instance = [
            instance[descriptive_header.index(key)]
            for key in descriptive_header
        ]
        print("Mapped Instance for Prediction:", mapped_instance)

        # Use the classifier to predict
        prediction = tree_classifier.predict([mapped_instance])[0]
        print("Prediction result:", prediction)
        return prediction
    except Exception as e:
        print(f"Error predicting position: {e}")
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
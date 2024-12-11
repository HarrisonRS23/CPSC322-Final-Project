import os
import pickle
from flask import Flask, render_template, request, jsonify
from mysklearn.myclassifiers import MyDecisionTreeClassifier
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pickled decision tree and header
def load_tree():
    try:
        with open("soccer_tree.p", "rb") as infile:
            header, tree = pickle.load(infile)
        return header, tree
    except FileNotFoundError:
        logger.error("Error: The file 'soccer_tree.p' was not found.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading tree: {e}")
        return None, None

# Load the tree and header
header, soccer_tree = load_tree()
if header is None or soccer_tree is None:
    logger.error("Failed to load decision tree or header. Exiting...")
    exit(1)

logger.info(f"Loaded Header: {header}")

# Attribute mapping for descriptive names
attribute_mapping = {
    "height_cm": "att0",
    "positions": "att1",
    "skill_moves": "att2",
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
descriptive_header = list(attribute_mapping.keys())
logger.info(f"Mapped Header (Descriptive): {descriptive_header}")

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
            instance = [request.form.get(attr, "").strip() for attr in descriptive_header]
            if not all(instance):
                raise ValueError("Missing or invalid input data.")
            prediction = predict_player_position(instance)
        except Exception as e:
            logger.error(f"Error processing form input: {e}")
            prediction = "Invalid input"
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            # Collect query parameters based on descriptive header
            instance = [request.args.get(attr, "").strip() for attr in descriptive_header]
        elif request.method == "POST":
            # Collect JSON payload for POST request
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid or missing JSON payload"}), 400
            instance = [data.get(attr, "").strip() for attr in descriptive_header]

        logger.info(f"Instance received: {instance}")

        if None in instance or "" in instance:  # Ensure all attributes are present
            return jsonify({"error": "Missing or invalid input data"}), 400

        # Predict the position
        prediction = predict_player_position(instance)
        if prediction is not None:
            return jsonify({"prediction": prediction}), 200
        return jsonify({"error": "Prediction could not be made"}), 400
    except Exception as e:
        logger.error(f"Error in prediction route: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def predict_player_position(instance):
    """
    Predict the player's position using the decision tree.
    """
    try:
        # Convert descriptive attributes to generic names for prediction
        mapped_instance = [
            instance[descriptive_header.index(attr)]
            for attr in descriptive_header
        ]
        logger.info(f"Mapped Instance for Prediction: {mapped_instance}")

        # Use the classifier to predict
        prediction = tree_classifier.predict([mapped_instance])[0]
        logger.info(f"Prediction result: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error predicting position: {e}")
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

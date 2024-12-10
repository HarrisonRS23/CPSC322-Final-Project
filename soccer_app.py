import pickle
from flask import Flask, request, jsonify
from random import sample, seed
# some useful mysklearn package import statements and reloads
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

# uncomment once you paste your mypytable.py into mysklearn package
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable

# uncomment once you paste your myclassifiers.py into mysklearn package
import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

# Initialize the KNeighborsClassifier
knn_classifier = MyKNeighborsClassifier()
dummy_classifier = MyDummyClassifier()
naive_class = MyNaiveBayesClassifier()
tree_classifier = MyDecisionTreeClassifier()

app = Flask(__name__)

# Global variables for preprocessed FIFA data
table = None

# Preprocess FIFA data
def preprocess_fifa_data():
    global table
    seed(42)  # For reproducibility
    table = MyPyTable().csv_to_mypytable("input_file/fifa_players.csv")

    # Filter rows where overall_rating is < 70
    rating_column = table.get_column("overall_rating")
    indexes_to_drop = [index for index, rating in enumerate(rating_column) if int(rating) < 70]
    table.drop_rows(indexes_to_drop)

    # Remove unnecessary columns
    columns_to_remove = [
        "name", "full_name", "birth_date", "age", "weight_kgs", "nationality",
        "overall_rating", "potential", "value_euro", "wage_euro", "preferred_foot",
        "international_reputation(1-5)", "weak_foot(1-5)", "body_type", "release_clause_euro",
        "national_team", "national_rating", "national_team_position", "national_jersey_number",
        "heading_accuracy", "volleys", "curve", "sprint_speed", "reactions", "balance",
        "jumping", "strength", "aggression", "penalties", "composure", "sliding_tackle"
    ]
    for column in columns_to_remove:
        if column in table.column_names:
            table.remove_column(column)

    # Discretize positions
    positions = table.get_column("positions")
    discretized_positions = [myutils.classify_position(position) for position in positions]
    table.add_column("discretized_position", discretized_positions)

    # Downsample positions
    goalkeepers = [row for row in table.data if row[-1] == "Goalkeeper"]
    defenders = [row for row in table.data if row[-1] == "Defender"]
    forwards = [row for row in table.data if row[-1] == "Forward"]
    midfielders = [row for row in table.data if row[-1] == "Midfielder"]

    gk_size = len(goalkeepers)
    defenders_downsampled = sample(defenders, min(gk_size, len(defenders)))
    forwards_downsampled = sample(forwards, min(gk_size, len(forwards)))
    midfielders_downsampled = sample(midfielders, min(gk_size, len(midfielders)))

    # Combine balanced data
    balanced_data = goalkeepers + defenders_downsampled + forwards_downsampled + midfielders_downsampled
    table.data = balanced_data


@app.route("/")
def index():
    return "<h1>Welcome to the FIFA Player Predictor App</h1>", 200


@app.route("/predict")
def predict():
    try:
        # Get query parameters
        height_cm = float(request.args.get("height_cm", 0))
        positions = request.args.get("positions", "")
        skill_moves = int(request.args.get("skill_moves", 0))
        crossing = int(request.args.get("crossing", 0))

        # Mock prediction logic
        # Use a simple rule-based system for demo purposes
        if height_cm > 190 and "Goalkeeper" in positions:
            prediction = "Goalkeeper"
        elif skill_moves > 3 and crossing > 70:
            prediction = "Forward"
        else:
            prediction = "Midfielder"

        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    preprocess_fifa_data()  # Preprocess the FIFA data before running the app
    app.run(host="0.0.0.0", port=5001, debug=True)
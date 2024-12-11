import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the pickled decision tree and header
def load_tree():
    try:
        with open("soccer_tree.p", "rb") as infile:
            header, tree = pickle.load(infile)
        return header, tree
    except Exception as e:
        print(f"Error loading tree: {e}")
        return None, None

header, soccer_tree = load_tree()

@app.route("/", methods=["GET", "POST"])
def index_page():
    prediction = ""
    if request.method == "POST":
        try:
            # Dynamically parse inputs from the form
            instance = [request.form.get(attr, "") for attr in header]
            prediction = predict_player_position(instance)
        except Exception as e:
            print(f"Error processing form input: {e}")
            prediction = "Invalid input"
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        instance = [request.args.get(attr, "") for attr in header]
        prediction = predict_player_position(instance)
        if prediction is not None:
            return jsonify({"prediction": prediction}), 200
        return "Error making prediction", 400
    except Exception as e:
        print(f"Error in prediction API: {e}")
        return "Error making prediction", 400

def tdidt_classifier(tree, header, instance):
    """
    Classify an instance using the TDIDT tree.
    """
    if tree[0] == "Attribute":
        attribute_index = header.index(tree[1])
        test_value = instance[attribute_index]
        for value_subtree in tree[2:]:
            if value_subtree[1] == test_value:
                return tdidt_classifier(value_subtree[2], header, instance)
    elif tree[0] == "Leaf":
        return tree[1]
    return None

def predict_player_position(instance):
    """
    Predict the player's position using the decision tree.
    """
    try:
        return tdidt_classifier(soccer_tree, header, instance)
    except Exception as e:
        print(f"Error predicting position: {e}")
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
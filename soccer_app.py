import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the soccer decision tree and header once at startup
def load_tree():
    try:
        with open("soccer_tree.p", "rb") as infile:  # Ensure this matches your pickled tree file
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
        # Parse form inputs dynamically from the header
        try:
            instance = [request.form.get(attr, "") for attr in header]
            prediction = predict_player_position(instance)
        except Exception as e:
            print(f"Error processing form input: {e}")
            prediction = "Invalid input"
    print("Prediction:", prediction)
    return render_template("index.html", prediction=prediction)


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Parse query string inputs dynamically from the header
        instance = [request.args.get(attr, "") for attr in header]
        prediction = predict_player_position(instance)
        if prediction is not None:
            result = {"prediction": prediction}
            return jsonify(result), 200
        else:
            return "Error making prediction", 400
    except Exception as e:
        print(f"Error in prediction API: {e}")
        return "Error making prediction", 400


def tdidt_classifier(tree, header, instance):
    """
    Recursive function to classify an instance using the decision tree.
    """
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        test_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == test_value:
                return tdidt_classifier(value_list[2], header, instance)
    elif info_type == "Leaf":
        return tree[1]  # Return the class label from the leaf
    else:
        return None


def predict_player_position(unseen_instance):
    """
    Predicts the position of a soccer player using the decision tree.
    """
    try:
        return tdidt_classifier(soccer_tree, header, unseen_instance)
    except Exception as e:
        print(f"Error predicting position: {e}")
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
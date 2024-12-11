import os
import pickle

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        # Get soccer attributes from the form
        height_cm = request.form["height_cm"]
        short_passing = request.form["short_passing"]
        vision = request.form["vision"]
        crossing = request.form["crossing"]
        
        # Create an instance based on input
        instance = [height_cm, short_passing, vision, crossing]
        prediction = predict_player_position(instance)
    print("Prediction:", prediction)
    return render_template("index.html", prediction=prediction)

@app.route('/predict', methods=["GET"])
def predict():
    # Parse soccer attributes from query string
    height_cm = request.args.get("height_cm")
    short_passing = request.args.get("short_passing")
    vision = request.args.get("vision")
    crossing = request.args.get("crossing")
    
    # Create an instance based on input
    instance = [height_cm, short_passing, vision, crossing]
    prediction = predict_player_position(instance)
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

# Recursive TDIDT classifier
def tdidt_classifier(tree, header, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        test_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == test_value:
                return tdidt_classifier(value_list[2], header, instance)
    else:  # info_type == "Leaf"
        leaf_label = tree[1]
        return leaf_label

# Prediction function for soccer players
def predict_player_position(unseen_instance):
    # Load the soccer decision tree from the pickle file
    infile = open("soccer_tree.p", "rb")  # Replace with the correct pickle file name
    header, soccer_tree = pickle.load(infile)
    infile.close()
    try:
        return tdidt_classifier(soccer_tree, header, unseen_instance)
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
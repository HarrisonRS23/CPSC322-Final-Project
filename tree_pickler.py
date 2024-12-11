import pickle
from mysklearn.myclassifiers import MyDecisionTreeClassifier

# Step 1: Load and preprocess your soccer dataset
def load_soccer_data():
    """
    Function to load and preprocess the soccer dataset.
    Replace this with your actual dataset loading and preprocessing logic.
    """
    # Example header and dataset
    header = ["height_cm", "short_passing", "vision", "crossing", "position"]
    data = [
        [180, "high", "high", "medium", "Midfielder"],
        [175, "low", "medium", "low", "Defender"],
        [170, "high", "low", "high", "Forward"],
        [165, "medium", "low", "medium", "Goalkeeper"],
        # Add more rows of data here
    ]

    # Separate features and labels
    X_train = [row[:-1] for row in data]  # All columns except the last one
    y_train = [row[-1] for row in data]  # Only the last column
    return header, X_train, y_train

# Step 2: Create and train the decision tree
def build_and_pickle_decision_tree():
    header, X_train, y_train = load_soccer_data()
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.fit(X_train, y_train)

    # Save the decision tree and header to a pickle file
    with open("soccer_tree.p", "wb") as outfile:
        pickle.dump((header, tree_classifier.tree), outfile)

    print("Decision tree has been pickled and saved as 'soccer_tree.p'")

# Step 3: Run the function to build and save the tree
if __name__ == "__main__":
    build_and_pickle_decision_tree()
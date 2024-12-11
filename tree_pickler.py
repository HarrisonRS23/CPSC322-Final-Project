import pickle
from mysklearn.myclassifiers import MyDecisionTreeClassifier

def load_soccer_data():
    """
    Function to load and preprocess the soccer dataset.
    """
    header = ["height_cm", "short_passing", "vision", "crossing", "position"]
    data = [
        [180, "high", "high", "medium", "Midfielder"],
        [175, "low", "medium", "low", "Defender"],
        [170, "high", "low", "high", "Forward"],
        [165, "medium", "low", "medium", "Goalkeeper"],
    ]

    X_train = [row[:-1] for row in data]  # Features
    y_train = [row[-1] for row in data]  # Target labels
    return header, X_train, y_train

# Step 2: Create and train the decision tree
def build_and_pickle_decision_tree():
    header, X_train, y_train = load_soccer_data()
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.header = header  # Explicitly set the header
    tree_classifier.fit(X_train, y_train)

    # Save the decision tree and header to a pickle file
    with open("soccer_tree.p", "wb") as outfile:
        pickle.dump((header, tree_classifier.tree), outfile)

    print("Decision tree has been pickled and saved as 'soccer_tree.p'")

# Step 3: Run the function to build and save the tree
if __name__ == "__main__":
    build_and_pickle_decision_tree()
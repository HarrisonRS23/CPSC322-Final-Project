import pickle
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn import myutils

def load_and_preprocess_data():
    """
    Load and preprocess the FIFA dataset.
    """
    table = MyPyTable().csv_to_mypytable("input_file/fifa_players.csv")

    # Filter rows where overall_rating < 70
    rating_column = table.get_column("overall_rating")
    indexes_to_drop = [index for index, row in enumerate(rating_column) if int(row) < 70]
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
        table.remove_column(column)

    # Discretize positions
    positions = table.get_column("positions")
    discretized_positions = [myutils.classify_position(position) for position in positions]

    # Update the "positions" column in the table manually
    position_index = table.column_names.index("positions")
    for i in range(len(table.data)):
        table.data[i][position_index] = discretized_positions[i]

    # Specify columns for training
    columns_to_include = [
        "height_cm", "positions", "skill_moves(1-5)", "crossing", "finishing",
        "short_passing", "dribbling", "freekick_accuracy", "long_passing", "ball_control",
        "acceleration", "agility", "shot_power", "stamina", "long_shots", "interceptions",
        "positioning", "vision", "marking", "standing_tackle"
    ]
    column_indices = [table.column_names.index(col) for col in columns_to_include]

    # Extract features and target
    combined_list = [[row[idx] for idx in column_indices] for row in table.data]
    target = discretized_positions  # Target is already discretized
    return columns_to_include, combined_list, target

def build_and_pickle_decision_tree():
    """
    Build and pickle the decision tree using the FIFA dataset.
    """
    header, X_train, y_train = load_and_preprocess_data()
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.header = header  # Set the header explicitly
    tree_classifier.fit(X_train, y_train)

    # Save the decision tree and header to a pickle file
    with open("soccer_tree.p", "wb") as outfile:
        pickle.dump((header, tree_classifier.tree), outfile)

    print("Decision tree has been pickled and saved as 'soccer_tree.p'")

if __name__ == "__main__":
    build_and_pickle_decision_tree()
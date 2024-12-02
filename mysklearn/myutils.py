"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #7
11/20/24

Description: This is a data science story about evaluating classifiers using different performance matrixs. This file
holds my general usage utility functions.
"""
import numpy as np  # use numpy's random number generation
from mysklearn import myevaluation
from tabulate import tabulate


def shuffle(list1, list2, random_state):
    """
    Shuffle two lists while maintaining their parallel structure.

    Args:
        list1 (list): First list to shuffle.
        list2 (list): Second list to shuffle, corresponding to list1.
        random_state (int or None): Seed for random number generation for reproducibility.

    Returns:
        tuple: Two shuffled lists.
    """
    indices = list(range(len(list1)))
    non_shuffled_indices = indices
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)

    np.random.shuffle(indices)
    if non_shuffled_indices == indices:
        np.random.shuffle(indices)
    shuffled_list1 = [list1[i] for i in indices]
    shuffled_list2 = [list2[i] for i in indices]
    return shuffled_list1, shuffled_list2


def random_subsample(
        clf,
        X,
        y,
        k_sub_samples=10,
        test_size=0.33,
        discretizer=None):
    """Perform a random subsample for either k-NN or Dummy classifier based on the provided k value.

    Args:
        clf: The classifier to be used (e.g., k-NN or Dummy classifier).
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target values (parallel to X).
        k_sub_samples (int): Number of subsamples to average over.
        test_size (float): Ratio of test data size to total data.
        discretizer (optional): Discretizer function or object to preprocess target values.

    Returns:
        tuple: A tuple containing (average_accuracy, average_error_rate).
    """
    total_accuracy = 0
    total_error_rate = 0

    for _ in range(k_sub_samples):
        # Split the data
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate accuracy and error rate
        accuracy = myevaluation.accuracy_score(y_test, y_pred, normalize=True)
        error_rate = 1 - accuracy

        # Accumulate results
        total_accuracy += accuracy
        total_error_rate += error_rate

    # Calculate the average accuracy and error rate
    average_accuracy = total_accuracy / k_sub_samples
    average_error_rate = total_error_rate / k_sub_samples

    return average_accuracy, average_error_rate


def cross_val_predict(
        X,
        y,
        classifier,
        k=10,
        stratify=True,
        random_state=0,
        shuffle=False):
    """
    Perform k-fold cross-validation multiple times to evaluate classifier performance.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target y values (parallel to X).
        classifier (obj): Classifier instance with fit and predict methods (e.g., MyKNeighborsClassifier or MyDummyClassifier).
        k (int): Number of times to repeat the entire cross-validation process.
        stratify (bool): Whether to use stratified k-fold cross-validation.
        random_state (int, optional): Random seed for reproducibility.
        shuffle (bool): If True, shuffle the data before splitting.

    Returns:
        tuple: Overall mean accuracy and mean error rate across all k rounds of cross-validation.
    """
    all_accuracies = []
    n_splits = 5

    for i in range(k):
        if stratify:
            folds = myevaluation.stratified_kfold_split(
                X, y, random_state=random_state, shuffle=shuffle)
        else:
            folds = myevaluation.kfold_split(
                X, random_state=random_state, shuffle=shuffle)

        fold_accuracies = []

        for train_indices, test_indices in folds:
            # Split data according to current fold
            X_train = [X[index] for index in train_indices]
            X_test = [X[index] for index in test_indices]
            y_train = [y[index] for index in train_indices]
            y_test = [y[index] for index in test_indices]

            # Train and predict with classifier
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            # Calculate accuracy and append to fold_accuracies
            accuracy = myevaluation.accuracy_score(
                y_test, y_pred, normalize=True)
            fold_accuracies.append(accuracy)

        # Calculate mean accuracy for this round and add to all_accuracies
        all_accuracies.append(sum(fold_accuracies) / n_splits)

    # Calculate overall mean accuracy and error rate across all k rounds
    overall_mean_accuracy = sum(all_accuracies) / k
    overall_error_rate = 1 - overall_mean_accuracy
    return overall_mean_accuracy, overall_error_rate


def check_index(val, labels):
    for i, label in enumerate(labels):
        if label == val:  # Compare the label to val
            return i
    return -1  # Return -1 if the label is not found


def bootstrap_method(
        X,
        y,
        classifier,
        k=10,
        n_samples=None,
        random_state=None):
    """
    Perform bootstrap sampling to evaluate classifier performance.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target y values (parallel to X).
        classifier (obj): Classifier instance (e.g., MyKNeighborsClassifier or MyDummyClassifier).
        k (int): Number of bootstrap iterations.
        n_samples (int): Number of samples in each bootstrap sample. Defaults to size of X.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Mean accuracy and mean error rate across bootstrap samples.
    """
    accuracies = []

    for i in range(k):
        # Generate bootstrapped sample and out-of-bag sample
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(
            X, y, n_samples=n_samples, random_state=random_state)

        # Train and predict with classifier on out-of-bag sample
        classifier.fit(X_sample, y_sample)
        y_pred = classifier.predict(X_out_of_bag)

        # Calculate accuracy for this bootstrap iteration
        accuracy = myevaluation.accuracy_score(
            y_out_of_bag, y_pred, normalize=True)
        accuracies.append(accuracy)

        # Update random_state to ensure different sampling in next iteration
        if random_state is not None:
            random_state += 1

    # Calculate mean accuracy and error rate across all bootstrap samples
    mean_accuracy = sum(accuracies) / k
    error_rate = 1 - mean_accuracy
    return mean_accuracy, error_rate


def compute_confusion_matrices(
        X,
        y,
        classifier,
        labels,
        n_splits=10,
        random_state=None,
        stratified=False):
    """
    Perform k-fold cross-validation, compute confusion matrices for each fold,
    and return the cumulative confusion matrix.

    Args:
        X (list of list of obj): The list of instances (samples).
        y (list of obj): The target y values (parallel to X).
        classifier (obj): Classifier instance (e.g., MyKNeighborsClassifier or MyDummyClassifier).
        labels (list of str): The list of all possible target y labels for the confusion matrix.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        stratified (bool): Whether to use stratified k-fold split or regular k-fold split.

    Returns:
        list of list of int: Cumulative confusion matrix across all folds.
    """
    # Initialize cumulative confusion matrix
    cumulative_matrix = [[0 for _ in labels] for _ in labels]

    # Choose cross-validation method
    if stratified:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits, random_state=random_state)
    else:
        folds = myevaluation.kfold_split(
            X, y, n_splits=n_splits, random_state=random_state)

    # Perform cross-validation and compute confusion matrix for each fold
    for train_idx, test_idx in folds:
        # Split data into training and testing sets for this fold
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # Train classifier and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compute confusion matrix for this fold
        fold_matrix = myevaluation.confusion_matrix(y_test, y_pred, labels)

        # Accumulate fold_matrix into cumulative_matrix
        for i in range(len(labels)):
            for j in range(len(labels)):
                cumulative_matrix[i][j] += fold_matrix[i][j]

    return cumulative_matrix


def compute_euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors.

    Args:
        v1 (list or np.array): First input vector.
        v2 (list or np.array): Second input vector.

    Returns:
        float: The Euclidean distance between the two vectors.
    """
    return np.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))


def compute_distance(v1, v2):
    if v1 == v2:
        return 0
    return 1


def get_top_k_instances(row_distances, k):
    """Selects the top K instances based on the smallest distances.

    Args:
        row_distances (list of tuples): List of tuples containing distances and corresponding indices.
        k (int): Number of top instances to retrieve.

    Returns:
        list: List of the top K instances (indices).
    """

    row_distances.sort()  # Sorts in-place, so the closest distances come first
    top_k = []
    # Loop through the sorted list and get the top k instances
    for i in range(k):
        # Append the second element (the corresponding row/index/label) of the
        # sorted distances
        # Assuming row is the second element in the sublist
        top_k.append(row_distances[i])
    return top_k


def select_class_label(top_k_indices, y_train):
    """Selects the class label based on the top K neighbors.

    Args:
        top_k_indices (list of int): Indices of the top K instances in the training set.
        y_train (list): The labels corresponding to the training instances.

    Returns:       str: The predicted class label.
    """
    # Create a list to store the labels of the top K neighbors
    top_k_labels = [y_train[i] for i in top_k_indices]

    # Find the most frequent label among the top K labels
    # Returns the label with the highest frequency
    return max(set(top_k_labels), key=top_k_labels.count)


def combine(*lists):
    """
    Combine multiple lists into a single list of lists, where each inner list contains
    corresponding elements from the input lists.

    Args:
        *lists: Any number of lists to combine.

    Returns:
        list: A list containing inner lists, each with elements from the input lists
              at the same index.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    # Check if all lists have the same length
    if not all(len(lst) == len(lists[0]) for lst in lists):
        raise ValueError("All input lists must have the same length.")

    # Use zip to combine the lists into a list of lists
    combined = [list(elements) for elements in zip(*lists)]
    return combined


def cross_val_confusion_matrix_aggregated(
        classifier,
        X,
        y,
        n_splits=10,
        stratified=True,
        random_state=None):
    """
    Compute a single, aggregated confusion matrix across all folds in k-fold cross-validation.
    """
    # Define labels based on unique values in y
    labels = sorted(list(set(y)))
    num_classes = len(labels)

    # Initialize the aggregated confusion matrix with zeros
    aggregated_confusion_mat = [[0] * num_classes for _ in range(num_classes)]

    # Use stratified or standard k-fold cross-validation
    if stratified:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits, random_state=random_state)
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits, random_state=random_state)

    # For each fold, train, predict, and aggregate the confusion matrix
    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # Train and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compute fold confusion matrix and aggregate it
        fold_confusion_mat = myevaluation.confusion_matrix(
            y_test, y_pred, labels)
        for i in range(num_classes):
            for j in range(num_classes):
                aggregated_confusion_mat[i][j] += fold_confusion_mat[i][j]

    return aggregated_confusion_mat


def compute_matrix(
        model,
        features,
        labels,
        num_folds=10,
        use_stratified_split=False):
    """
    Perform k-fold cross-validation for the given model and dataset.

    Parameters:
    model : classifier object
        The model with fit() and predict() methods.
    features : list of lists
        The dataset containing the feature values.
    labels : list
        The target labels (already discretized or categorized).
    num_folds : int
        The number of folds for cross-validation.
    use_stratified_split : bool
        If True, apply stratified k-fold splitting to preserve label distributions.

    Returns:
    list: Accumulated confusion matrix from all folds.
    """
    # Choose the appropriate split function based on the stratified flag
    if use_stratified_split:
        fold_splits = myevaluation.stratified_kfold_split(
            features, labels, num_folds, random_state=42)
    else:
        fold_splits = myevaluation.kfold_split(
            features, n_splits=num_folds, random_state=42)

    # Initialize the confusion matrix (for binary classification, 2x2 matrix)
    # Format: [[TP, FP], [FN, TN]]
    accumulated_confusion_matrix = [[0, 0], [0, 0]]

    # Cross-validation loop
    for train_indices, test_indices in fold_splits:
        X_train = [features[i] for i in train_indices]
        y_train = [labels[i] for i in train_indices]
        X_test = [features[i] for i in test_indices]
        y_test = [labels[i] for i in test_indices]

        # Train the model and make predictions
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Generate confusion matrix for the current fold
        fold_confusion_matrix = myevaluation.confusion_matrix(
            y_test, predictions, labels=['H', 'A'])

        # Accumulate the confusion matrix from this fold into the total
        # confusion matrix
        for i in range(2):  # Binary classification, so 2 classes (yes/no)
            for j in range(2):
                accumulated_confusion_matrix[i][j] += fold_confusion_matrix[i][j]

    # Return the final accumulated confusion matrix
    return accumulated_confusion_matrix


def calculate_class_statistics(matrix):
    """Calculate the total counts and recognition percentage for each class.

    Returns:
    list [int, float]: totals (class counts) and recognition percentages for each class
    """
    class_totals = [sum(row)
                    for row in matrix]  # Total instances for each class
    recognition_percentages = [
        (matrix[i][i] / class_totals[i] * 100 if class_totals[i]
         > 0 else 0)  # Recognition percentage for each class
        for i in range(len(matrix))
    ]
    return class_totals, recognition_percentages


def print_matrix(confusion_matrix, labels):
    """Displays a confusion matrix in a readable table format using tabulate."""
    print("===========================================")
    print("True vs Predicted Classifications")
    headers = ['Actual Class', 'Class 1', 'Class 2']

    # Prepare the data for the matrix table
    table_data = []
    for i, label in enumerate(labels):
        table_data.append(
            [label, confusion_matrix[i][0], confusion_matrix[i][1]])

    # Print the formatted confusion matrix
    print(tabulate(table_data, headers=headers, tablefmt="github"))


def group_by(data, key):
    """Group a list of dictionaries by a specified key.

    Args:
        data (list of dict): The data to group, where each item is a dictionary.
        key (str): The key to group by, which should exist in each dictionary.

    Returns:
        dict: A dictionary where each key is a unique value from the specified key in data,
              and each value is a list of dictionaries that share that key's value.
    """
    grouped_data = {}

    for item in data:
        # Get the value of the specified key for this item
        group_key = item[key]

        # Initialize the group if not already present
        if group_key not in grouped_data:
            grouped_data[group_key] = []

        # Append the current item to the group
        grouped_data[group_key].append(item)

    return grouped_data

# PA7


def select_attribute(instances, attributes, header):
    """
    Select the attribute with the smallest weighted entropy (Enew).

    Args:
        instances (list of list): The dataset.
        attributes (list of str): List of attribute names.
        header (list of str): List of all column names in the dataset.

    Returns:
        str: The best attribute for splitting.
    """
    best_attribute = None
    min_entropy = float('inf')

    for attribute in attributes:
        # Map attribute name to its index
        attribute_index = header.index(attribute)

        # Partition instances by the current attribute
        partitions = partition_instances(instances, attribute_index, attribute)

        # Calculate weighted entropy for the attribute
        total_instances = len(instances)
        weighted_entropy = 0.0

        for partition in partitions.values():
            partition_entropy = calculate_entropy(partition)
            weighted_entropy += (len(partition) /
                                 total_instances) * partition_entropy

        # Keep track of the attribute with the smallest entropy
        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            best_attribute = attribute
        elif weighted_entropy == min_entropy:
            # Break ties lexicographically
            if best_attribute is None or attribute < best_attribute:
                best_attribute = attribute

    return best_attribute


def calculate_entropy(instances):
    """
    Calculate the entropy of a dataset.

    Args:
        instances (list of list): A subset of the dataset where the last column is the class label.

    Returns:
        float: The entropy of the dataset.
    """
    class_counts = {}
    for row in instances:
        class_label = row[-1]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    total_instances = len(instances)
    probabilities = np.array(list(class_counts.values())) / total_instances

    # Use numpy to calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def all_same_class(instances):
    """
    Function to determine if all instances are apart of the same class label
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True


def extract_domains(X_train, column_names):
    """
    Extracts the unique domain (set of unique values) for each column in the input data
    and returns it in the format of a dictionary.

    Args:
        X_train (list of list): A 2D list where each inner list is a row of data.
        column_names (list of str): A list of column names corresponding to the data.

    Returns:
        dict: A dictionary where keys are column names and values are lists of unique values.
    """
    if not X_train or not column_names:
        return {}  # Return an empty dictionary if input is empty

    attribute_domains = {}

    # Transpose the data to process each column
    for col_idx, col_name in enumerate(column_names):
        column = [row[col_idx] for row in X_train]
        # Extract unique values and sort them
        domain = sorted(list(set(column)))
        attribute_domains[col_name] = domain

    return attribute_domains


def get_header(X_train):
    """
    Generates a header list of attribute names based on the number of attributes in X_train.

    Parameters:
    X_train (list of lists): The training dataset where each entry is a list of attributes.

    Returns:
    list: A list of attribute names in the format ['attr0', 'attr1', ..., 'attrN'].
    """
    # Determine the number of attributes in the first row
    length = len(X_train[0])
    # Create a list of attribute names
    header = [f"att{i}" for i in range(length)]
    return header


def majority_class(partition):
    """
    Determines the majority class in a dataset partition.

    Args:
        partition (list of lists): Rows of data where the last column is the class label.

    Returns:
        str: The majority class label.
    """

    # Extract class labels (last column in each row)
    class_labels = [row[-1] for row in partition]

    # Count occurrences of each class
    label_counts = {}
    for label in class_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Return the class with the maximum count
    return max(label_counts, key=label_counts.get)


def partition_instances(instances, attribute_index, attribute_name=None):
    """
    Partition the dataset based on the values of a specified attribute.

    Args:
        instances (list of list): The dataset.
        attribute_index (int): The index of the attribute to partition by.
        attribute_name (str, optional): The name of the attribute (for debugging).

    Returns:
        dict: A dictionary where keys are unique attribute values and values are lists of instances.
    """
    partitions = {}
    for row in instances:
        attribute_value = row[attribute_index]
        if attribute_value not in partitions:
            partitions[attribute_value] = []
        partitions[attribute_value].append(row)
    return partitions


def perform_analysis(
        combined_list,
        target,
        knn_classifier,
        dummy_classifier,
        naive_class,
        tree_classifier):
    """
    Function to repeat
    """
    # Perform 10-fold cross-validation with stratified
    knn_accuracy, knn_error_rate = cross_val_predict(
        combined_list, target, knn_classifier, stratify=True, random_state=42)
    dummy_accuracy, dummy_error_rate = cross_val_predict(
        combined_list, target, dummy_classifier, stratify=True, random_state=42)
    naive_accuracy, naive_error_rate = cross_val_predict(
        combined_list, target, naive_class, stratify=True, random_state=42)
    tree_accuracy, tree_error_rate = cross_val_predict(
        combined_list, target, tree_classifier, stratify=True, random_state=42)

    print("(BONUS) Stratified 10-Fold Cross Validation")

    print("Accuracy and Error Rate")
    # Print results for kNN, Dummy, and Naive classifiers
    print(
        f"k Nearest Neighbors Classifier: accuracy = {
            knn_accuracy:.2f}, error rate = {
            knn_error_rate:.2f}")
    print(
        f"Dummy Classifier: accuracy = {
            dummy_accuracy:.2f}, error rate = {
            dummy_error_rate:.2f}")
    print(
        f"Naive Classifier: accuracy = {
            naive_accuracy:.2f}, error rate = {
            naive_error_rate:.2f}")
    print(
        f"Tree Classifier: accuracy = {
            tree_accuracy:.2f}, error rate = {
            tree_error_rate:.2f}")

    print("Precision, recall, and F1 measure")
    # Get predictions from both classifiers
    knn_predictions = knn_classifier.predict(combined_list)
    dummy_predictions = dummy_classifier.predict(combined_list)
    naive_predictions = naive_class.predict(combined_list)
    tree_predictions = tree_classifier.predict(combined_list)

    # Calculate metrics for all classifiers
    knn_recall = myevaluation.binary_recall_score(target, knn_predictions)
    dummy_recall = myevaluation.binary_recall_score(target, dummy_predictions)
    naive_recall = myevaluation.binary_recall_score(target, naive_predictions)
    tree_recall = myevaluation.binary_recall_score(target, tree_predictions)

    knn_precision = myevaluation.binary_precision_score(
        target, knn_predictions)
    dummy_precision = myevaluation.binary_precision_score(
        target, dummy_predictions)
    naive_precision = myevaluation.binary_precision_score(
        target, naive_predictions)
    tree_precision = myevaluation.binary_precision_score(
        target, tree_predictions)

    knn_f1 = myevaluation.binary_f1_score(target, knn_predictions)
    dummy_f1 = myevaluation.binary_f1_score(target, dummy_predictions)
    naive_f1 = myevaluation.binary_f1_score(target, naive_predictions)
    tree_f1 = myevaluation.binary_f1_score(target, tree_predictions)

    # Print the results in a condensed format
    print(
        f"kNN Classifier: recall = {
            knn_recall:.2f}, precision = {
            knn_precision:.2f}, F1 = {
                knn_f1:.2f}")
    print(
        f"Dummy Classifier: recall = {
            dummy_recall:.2f}, precision = {
            dummy_precision:.2f}, F1 = {
                dummy_f1:.2f}")
    print(
        f"Naive Classifier: recall = {
            naive_recall:.2f}, precision = {
            naive_precision:.2f}, F1 = {
                naive_f1:.2f}")
    print(
        f"Tree Classifier: recall = {
            tree_recall:.2f}, precision = {
            tree_precision:.2f}, F1 = {
                tree_f1:.2f}")

"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #7
11/20/24

Description: This is a data science story about evaluating classifiers using different performance matrixs. This file
holds my evaluation functions.
"""

import numpy as np  # use numpy's random number generation
from mysklearn import myutils
from tabulate import tabulate


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))  # unique values
    if pos_label is None:
        pos_label = labels[0]

    matrix = confusion_matrix(y_true, y_pred, labels)
    # Identify the index of the positive label
    pos_index = 0
    for index, val in enumerate(labels):
        if pos_label == val:
            pos_index = index

    # Calculate TP and FP
    tp = matrix[pos_index][pos_index]  # Position of TP
    fp = sum(matrix[row][pos_index] for row in range(
        len(labels)) if row != pos_index)  # sum all FP

    # Handle division by zero
    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))  # unique values
    if pos_label is None:
        pos_label = labels[0]

    matrix = confusion_matrix(y_true, y_pred, labels)
    # Identify the index of the positive label
    pos_index = 0
    for index, val in enumerate(labels):
        if pos_label == val:
            pos_index = index

    # Calculate TP and FP
    tp = matrix[pos_index][pos_index]  # Position of TP
    fn = sum(matrix[pos_index][row] for row in range(
        len(labels)) if row != pos_index)  # sum all FN

    # Handle division by zero
    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision_val = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall_val = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision_val + recall_val == 0.0:
        return 0.0

    return (2 * (precision_val * recall_val)) / (precision_val + recall_val)


"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #5
10/28/24
I completed the bonus

Description: This is a data science story about evaluating classifiers using different performance matrixs. This
file includes my evaluation functions.
"""


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        X, y = myutils.shuffle(X, y, random_state)

    if isinstance(test_size, float):
        # test_size of randomized table is train, rest is test
        split_index = int((1 - test_size) * len(X))
        return X[0:split_index], X[split_index:], y[0:split_index], y[split_index:]

    split_index = len(X) - test_size
    return X[0:split_index], X[split_index:], y[0:split_index], y[split_index:]


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    # Create a list of indices
    indices = list(range(len(X)))
    if shuffle:
        np.random.shuffle(indices)

    # Calculate fold sizes
    fold_sizes = [len(X) // n_splits + (1 if i < len(X) %
                                        n_splits else 0) for i in range(n_splits)]

    # Step 2: Split indices into folds
    folds = []
    start_index = 0
    for size in fold_sizes:
        fold = indices[start_index:start_index + size]
        folds.append(fold)
        start_index += size

    # Generate the train/test sets
    final_folds = []
    for i in range(n_splits):
        test = folds[i]
        train = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        final_folds.append((train, test))

    return final_folds


import numpy as np  # Ensure NumPy is imported for random operations

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Seed the random generator for reproducibility, if required
    if random_state is not None:
        np.random.seed(random_state)

    # Create a dictionary to map each unique label to its corresponding indices
    label_indices = {}
    for id, label in enumerate(y):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(id)

    # Shuffle each label group if shuffle is True
    if shuffle:
        for indices in label_indices.values():
            np.random.shuffle(indices)
            np.random.shuffle(indices)
            np.random.shuffle(indices)

    # Initialize folds
    folds = [[] for _ in range(n_splits)]

    # Distribute each label's indices across the folds
    for label, indices in label_indices.items():
        for i, id in enumerate(indices):
            folds[i % n_splits].append(id)

    # Generate training and testing sets for each fold
    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [id for j in range(
            n_splits) if j != i for id in folds[j]]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds



def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if n_samples is None:
        n_samples = len(X)
    if random_state is not None:
        np.random.seed(random_state)

    # Step 1: Generate bootstrap sample indices
    # Sampling with replacement
    indices = np.random.randint(0, len(X), size=n_samples)
    unique_indices = set(range(len(X))) - set(indices)  # OOB sample indices

    # Step 2: Create X_sample and X_out_of_bag
    X_sample = [X[i] for i in indices]
    X_out_of_bag = [X[i] for i in unique_indices]

    # Step 3: If y is provided, create y_sample and y_out_of_bag
    if y is not None:
        y_sample = [y[i] for i in indices]
        y_out_of_bag = [y[i] for i in unique_indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in labels]
              for _ in labels]  # Create matrix in form with filled in zeros
    for count, true_val in enumerate(y_true):
        predicted_label = y_pred[count]
        true_index = myutils.check_index(
            true_val, labels)  # Find index for the true value
        # Find index for the predicted value
        pred_index = myutils.check_index(predicted_label, labels)
        if true_val == predicted_label:  # Check if correct prediction
            matrix[true_index][true_index] += 1  # Insert into matrix
        else:
            matrix[true_index][pred_index] += 1
    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:  # Compare values directly
            tp += 1
    if normalize:
        return tp / len(y_true)  # Use len(y_true) for normalization
    return tp


def classification_report(y_true, y_pred, labels=None, output_dict=False):
    if labels is None:
        labels = sorted(set(y_true))

    # Initialize dictionary to hold metrics
    report = {}

    # Calculate metrics per label
    for label in labels:
        tp = sum((yt == label and yp == label)
                 for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label)
                 for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label)
                 for yt, yp in zip(y_true, y_pred))
        support = sum(1 for yt in y_true if yt == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0.0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support
        }

    # Calculate macro and weighted averages
    total_support = sum(report[label]["support"] for label in labels)

    # Compute averages and round to two decimal places
    macro_precision = round(
        sum(report[label]["precision"] for label in labels) / len(labels), 2)
    macro_recall = round(sum(report[label]["recall"]
                         for label in labels) / len(labels), 2)
    macro_f1_score = round(sum(report[label]["f1-score"]
                           for label in labels) / len(labels), 2)

    weighted_precision = round(
        sum(
            report[label]["precision"] *
            report[label]["support"] for label in labels) /
        total_support,
        2)
    weighted_recall = round(sum(report[label]["recall"] *
                                report[label]["support"] for label in labels) /
                            total_support, 2)
    weighted_f1_score = round(
        sum(
            report[label]["f1-score"] *
            report[label]["support"] for label in labels) /
        total_support,
        2)

    # Add averages to the report
    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1_score,
        "support": total_support
    }
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1_score,
        "support": total_support
    }

    # Output as dict or formatted string
    if output_dict:
        return report
    else:
        headers = ["Label", "Precision", "Recall", "F1-score", "Support"]
        table_data = [[label,
                       f"{metrics['precision']:.2f}",
                       f"{metrics['recall']:.2f}",
                       f"{metrics['f1-score']:.2f}",
                       metrics["support"]] for label,
                      metrics in report.items()]
        return tabulate(table_data, headers=headers, tablefmt="github")

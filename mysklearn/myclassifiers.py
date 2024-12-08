from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import operator
import numpy as np
import random
from collections import Counter
from mysklearn import myutils
from collections import Counter
import numpy as np
import random

class MyRandomForestClassifier:
    def __init__(self, n_trees, max_features=5):
        """
        Initialize the Random Forest Classifier.

        Args:
            n_trees (int): Number of trees in the forest.
            max_features (int): Maximum number of features to consider when splitting a node.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []  # Stores trained decision trees
        self.tree_accuracies = []  # Stores validation accuracies of trees

    def fit(self, X, y):
        """
        Fit the Random Forest to the data.

        Args:
            X (list of list of obj): Training feature data.
            y (list of obj): Training labels.
        """
        X_train, y_train, X_val, y_val = self._train_test_split(X, y, test_size=0.33)
        bootstrap_samples = [self._bootstrap_sample(X_train, y_train) for _ in range(self.n_trees)]

        for X_boot, y_boot in bootstrap_samples:
            tree = MyDecisionTreeClassifier()
            selected_features = self._random_attribute_subset(len(X[0]), self.max_features)
            tree.fit(X_boot, y_boot)
            
            # Evaluate the tree's accuracy on the validation set
            accuracy = self._evaluate_tree(tree, X_val, y_val)
            self.trees.append(tree)
            self.tree_accuracies.append(accuracy)

        # Sort trees based on accuracy and keep only the most accurate
        sorted_trees = sorted(
            zip(self.trees, self.tree_accuracies), 
            key=lambda item: item[1], 
            reverse=True
        )
        self.trees, self.tree_accuracies = zip(*sorted_trees[:self.n_trees])
        self.trees = list(self.trees)
        self.tree_accuracies = list(self.tree_accuracies)

    def _train_test_split(self, X, y, test_size=0.33):
        """
        Split the data into training and test sets.

        Args:
            X (list of list of obj): Feature data.
            y (list of obj): Labels.
            test_size (float): Proportion of data to use for testing.

        Returns:
            tuple: Training and test sets.
        """
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(len(X) * (1 - test_size))
        X_train, y_train = [X[i] for i in indices[:split_idx]], [y[i] for i in indices[:split_idx]]
        X_test, y_test = [X[i] for i in indices[split_idx:]], [y[i] for i in indices[split_idx:]]
        return X_train, y_train, X_test, y_test

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample.

        Args:
            X (list of list of obj): Feature data.
            y (list of obj): Labels.

        Returns:
            tuple: Bootstrapped training data and labels.
        """
        n = len(X)
        indices = np.random.choice(range(n), n, replace=True)
        return [X[i] for i in indices], [y[i] for i in indices]

    def _random_attribute_subset(self, n_features, num_to_select):
        """
        Select a random subset of attributes.

        Args:
            n_features (int): Total number of features.
            num_to_select (int): Number of features to select.

        Returns:
            list of int: Indices of selected features.
        """
        all_features = list(range(n_features))
        np.random.shuffle(all_features)
        return all_features[:num_to_select]

    def _evaluate_tree(self, tree, X_val, y_val):
        """
        Evaluate a tree's accuracy on validation data.

        Args:
            tree (MyDecisionTreeClassifier): A decision tree classifier.
            X_val (list of list of obj): Validation feature data.
            y_val (list of obj): Validation labels.

        Returns:
            float: Accuracy of the tree.
        """
        predictions = tree.predict(X_val)
        return sum(pred == true for pred, true in zip(predictions, y_val)) / len(y_val) if len(y_val) > 0 else 0

    def predict(self, X):
        """
        Predict classes using the forest.

        Args:
            X (list of list of obj): Test feature data.

        Returns:
            list of obj: Predicted labels.
        """
        tree_predictions = [tree.predict(X) for tree in self.trees]
        tree_predictions = np.array(tree_predictions).T
        return [Counter(row).most_common(1)[0][0] for row in tree_predictions]
    
    
"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #7
11/20/24

Description: This is a data science story about evaluating classifiers using different performance matrixs. This file
holds my classifier functions.
"""


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # programatically exract header and domains
        self.header = myutils.get_header(X_train)
        self.attribute_domains = myutils.extract_domains(X_train, self.header)

        # lets stich together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # copy as python is pass by object reference
        available_attributes = self.header.copy()
        self.tree = self.tdidt(train, available_attributes)

    def partition_instances(self, instances, attribute):
        """
        Partitions a list of instances based on the domain of a specific attribute.

        Args:
            instances (list of list): The dataset to be partitioned, where each instance is a list of attribute values.
            att_index (int): The index of the attribute to partition by.
            att_domain (list): The domain (possible values) of the attribute at att_index.

        Returns:
            dict: A dictionary where keys are attribute values from the domain and values are lists of instances
                that match the corresponding attribute value.
        """
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain:  # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def tdidt(self, current_instances, available_attributes):
        """
        Top-Down Induction of Decision Trees (TDIDT) algorithm to construct a decision tree recursively.

        Args:
            current_instances (list of list): The current subset of data instances being processed. Each instance
                                            is a list of attribute values.
            available_attributes (list of str): The list of attributes available for splitting at this level.

        Returns:
            list: A nested tree structure representing the decision tree. The tree is a list where:
                - The first element is "Attribute" or "Value".
                - The second element is the name of the attribute or value.
                - Subtrees are nested lists under the corresponding attribute values.
        """

        # Select an attribute to split on
        split_attribute = myutils.select_attribute(
            current_instances, available_attributes, self.header)
        # print("splitting on:", split_attribute)

        # Get the index of the selected attribute
        split_attribute_index = self.header.index(split_attribute)

        # Copy and remove the split attribute to avoid modifying the original
        # list
        available_attributes = available_attributes.copy()
        available_attributes.remove(split_attribute)

        # Initialize the subtree with the split attribute
        tree = ["Attribute", split_attribute]

        # Partition the instances based on the split attribute
        partitions = self.partition_instances(
            current_instances, split_attribute)
        # print("partitions:", partitions)

        # Process each partition
        for att_value in sorted(partitions.keys()):  # Alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]

            # CASE 1: All class labels in the partition are the same
            if len(att_partition) > 0 and myutils.all_same_class(att_partition):
                # print("CASE 1")
                class_label = att_partition[0][-1]
                count_of_class_label = len(att_partition)
                total_count = len(current_instances)
                value_subtree.append(
                    ["Leaf", class_label, count_of_class_label, total_count])

            # CASE 2: No more attributes to split on
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # print("CASE 2")
                majority_class = myutils.majority_class(att_partition)
                count_of_class_label = sum(
                    1 for row in att_partition if row[-1] == majority_class)
                total_count = len(att_partition)
                value_subtree.append(
                    ["Leaf", majority_class, count_of_class_label, total_count])

            # CASE 3: No more instances in the partition
            elif len(att_partition) == 0:
                # print("CASE 3")
                majority_class = myutils.majority_class(current_instances)
                count_of_class_label = sum(
                    1 for row in current_instances if row[-1] == majority_class)
                total_count = len(current_instances)
                value_subtree.append(
                    ["Leaf", majority_class, count_of_class_label, total_count])

            # Recursive case: Split further
            else:
                # print("Recursing...")
                subtree = self.tdidt(att_partition, available_attributes)
                value_subtree.append(subtree)

            # Append the value subtree to the main tree
            tree.append(value_subtree)

        return tree

    def predict(self, X_test):
        """
        Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of obj): The list of testing samples.
                The shape of X_test is (n_test_samples, n_features).

        Returns:
            y_predicted (list of obj): The predicted target y values (parallel to X_test).
        """
        y_predicted = []  # Initialize an empty list to store predictions
        for instance in X_test:
            # Use the tdidt_predict method to predict for each instance
            prediction = self.tdidt_predict(self.tree, instance)
            y_predicted.append(prediction)
        return y_predicted

    def tdidt_predict(self, tree, instance):
        """
        Predicts the class label for a single instance using the given decision tree.

        Args:
            tree (list): A decision tree represented as a nested list structure.
                - The tree consists of "Attribute" and "Leaf" nodes.
                - "Attribute" nodes define the attribute to split on and contain subtrees for each value of the attribute.
                - "Leaf" nodes contain the predicted class label.
            instance (list): A single data instance (list of attribute values) to classify.

        Returns:
            obj: The predicted class label for the given instance.

        How It Works:
            - Recursively traverses the decision tree based on the attribute values in the instance.
            - Matches the instance's value for the current attribute to the corresponding subtree.
            - Stops at a "Leaf" node and returns the class label stored in the leaf.
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0]  # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1]  # class label

        # if we are here, we are at an Attribute
        # we need to match the instance's value for this attribute
        # to the appropriate subtree
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)

     # Function to collect paths from root to leaf nodes

    def collect_paths(self, node, path, paths):
        """
        Traverses the decision tree and collects all paths from the root to the leaf nodes.
        Each path is a list of attribute-value pairs, ending with the class label.

        Args:
            node (list): Current node being traversed.
            path (list): Current path being constructed.
            paths (list): List to store all paths from root to leaf.
        """
        if not node or len(node) == 0:
            return

        # Determine the type of node (e.g., Attribute, Value, Leaf)
        node_type = node[0]

        if node_type == "Attribute":
            # Add the attribute name to the path
            attribute = node[1]
            for branch in node[2:]:
                if branch[0] == "Value":
                    # Add the attribute value to the path
                    value = branch[1]
                    # Add as a tuple (attribute, value)
                    path.append((attribute, value))
                    if len(branch) > 2:
                        self.collect_paths(branch[2], path, paths)
                    path.pop()  # Backtrack after exploring the branch
        elif node_type == "Leaf":
            # Leaf node, append the class label to the path and save the path
            class_label = node[1]
            # Add the class label at the end
            paths.append(path + [class_label])

    # Wrapper function to get all paths from root to leaf nodes
    def paths(self):
        """
        Wrapper to collect all paths from the root to the leaf nodes in the tree.
        """
        all_paths = []
        self.collect_paths(self.tree, [], all_paths)
        return all_paths

    # Function to print decision rules
    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """
        Prints the decision rules from the tree in the format:
        "IF attr == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names (list or None): List of attribute names to use in the rules. Defaults to attribute indexes if None.
            class_name (str): The name of the class attribute in the rules.
        """
        all_paths = self.paths()  # Get all root-to-leaf paths

        # Generate rules
        for path in all_paths:
            rule = "IF "
            conditions = []

            # Iterate through the path, excluding the last element (class
            # label)
            for attr, val in path[:-1]:
                attribute = attribute_names[int(
                    attr[3:])] if attribute_names else attr
                conditions.append(f"{attribute} == {val}")

            # Add the class label at the end
            class_label = path[-1]
            rule += " AND ".join(conditions) + \
                f" THEN {class_name} = {class_label}"

            print(rule)

    # BONUS method

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass  # TODO: (BONUS) fix this

    """
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #6
11/7/24
I completed the bonus

Description: This is a data science story about evaluating classifiers using different performance matrixs. This file
holds my classifiers.
"""


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.header = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        combined_data = [X_train[i] + [y_train[i]]
                         for i in range(len(X_train))]
        grouped_data = myutils.group_by(combined_data, -1)

        self.priors = {}
        self.posteriors = {}
        labels = list(set(y_train))
        total_samples = len(y_train)

        # Track all possible values for each feature across the entire dataset
        feature_values = {feature_name: set() for feature_name in self.header}
        for instance in X_train:
            for feature_idx, feature_name in enumerate(self.header):
                feature_values[feature_name].add(instance[feature_idx])

        # Calculate priors for each label
        for label in labels:
            self.priors[label] = len(grouped_data[label]) / total_samples

        # Calculate posteriors for each feature within each label
        for class_label, instances in grouped_data.items():
            self.posteriors[class_label] = {}

            for feature_idx, feature_name in enumerate(self.header):
                # Initialize all possible values
                feature_counts = {
                    val: 0 for val in feature_values[feature_name]}

                # Count occurrences of each feature value within the class
                # label
                for instance in instances:
                    feature_value = instance[feature_idx]
                    feature_counts[feature_value] += 1

                # Convert counts to probabilities, ensuring floats
                total_feature_count = sum(feature_counts.values())
                feature_probs = {
                    val: float(count) /
                    total_feature_count for val,
                    count in feature_counts.items()}
                self.posteriors[class_label][feature_name] = feature_probs

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []  # List to store the predicted labels for each test instance

        # Iterate through each test instance
        for sample in X_test:
            class_probabilities = {}  # To hold the class probabilities for this sample

            # Iterate through each class (e.g., "on time", "late", etc.)
            for class_name, class_data in self.posteriors.items():
                # Start with the prior probability of the class
                prob_class = self.priors.get(
                    class_name, 1)  # Use priors initialized in fit

                # Iterate through features and compute the likelihood for each
                # feature
                for feature_idx, feature_value in enumerate(sample):
                    # Get the feature name from the header
                    feature_name = self.header[feature_idx]
                    if feature_name in class_data:
                        feature_probs = class_data[feature_name]
                        prob_feature_given_class = feature_probs.get(
                            feature_value, 0)
                        prob_class *= prob_feature_given_class  # Multiply by the feature likelihood

                # Store the class probability for this instance
                class_probabilities[class_name] = prob_class

            # Find the class with the highest probability for this sample
            predicted_class = max(
                class_probabilities,
                key=class_probabilities.get)

            # Append the predicted class to the results list
            y_predicted.append(predicted_class)

        return y_predicted


"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #5
10/28/24
I completed the bonus

Description: This is a data science story about evaluating classifiers using different performance matrixs.
"""


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            # Initialize regressor if not provided
            self.regressor = MySimpleLinearRegressor()
        # Fit the linear regression model to the data
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying the discretizer
        to the numeric predictions from the regressor.

        Args:
            X_test (list of list of numeridcc vals): The list of testing samples.
                The shape of X_test is (n_test_samples, n_features).
                Note that n_features for simple regression is 1, so each sample is a list
                with one element e.g. [[0], [1], [2]].

        Returns:
            y_predicted (list of obj): The predicted target y values (parallel to X_test).
        """
        return self.regressor.predict(X_test)


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test (list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances (list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices (list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        all_distances = []  # To hold distances for all test instances
        all_neighbor_indices = []  # To hold neighbor indices for all test instances

        for test_instance in X_test:
            row_indexes_dists = []

            # Calculate distance from each training instance to the test
            # instance
            for i, row in enumerate(self.X_train):
                dist = myutils.compute_distance(row, test_instance)
                row_indexes_dists.append((i, dist))

            # Sort by distance (the second element in the tuple)
            row_indexes_dists.sort(key=operator.itemgetter(-1))

            k = self.n_neighbors
            top_k = row_indexes_dists[:k]

            # Extract distances and indices from top_k
            # Use the distance part of the tuple
            distances = [dist[1] for dist in top_k]
            # Use the index part of the tuple
            indices = [dist[0] for dist in top_k]

            # Append the results for the current test instance
            all_distances.append(distances)
            all_neighbor_indices.append(indices)

        return all_distances, all_neighbor_indices  # Return the accumulated results

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted (list of obj): The predicted target y values (parallel to X_test)
        """
        # Get the distances and neighbor indices using the kneighbors method
        distances, neighbor_indices = self.kneighbors(X_test)

        y_predicted = []  # Initialize a list to hold predictions

        # Iterate through the neighbors for each test instance
        for i in range(len(X_test)):
            # Get the indices of the top K neighbors for the current test
            # instance
            top_k_instances = neighbor_indices[i]
            # Use your method to determine the predicted class label
            prediction = myutils.select_class_label(
                top_k_instances, self.y_train)
            y_predicted.append(prediction)  # Append the prediction to the list

        return y_predicted  # Return the list of predictions


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self, strategy="most_frequent"):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None
        self.strategy = strategy
        self.class_distibution = None

    def fit(self, X_train, y_train):
        """Fits the classifier to the training data.

        Args:
            X_train (list of list): The training feature data (not used).
            y_train (list): The training labels.
        """
        if self.strategy == "most_frequent":
            self.most_common_label = max(set(y_train), key=y_train.count)
        elif self.strategy == "stratified":
            labels, counts = np.unique(y_train, return_counts=True)
            self.class_distribution = {
                label: count /
                len(y_train) for label,
                count in zip(
                    labels,
                    counts)}

    def predict(self, X_test):
        """Predicts the class labels for the given test data.

        Args:
            X_test (list of list): The test feature data (not used).

        Returns:
            list: Predicted class labels.
        """
        if self.strategy == "most_frequent":
            return [self.most_common_label] * len(X_test)
        labels = list(self.class_distribution.keys())
        probabilities = list(self.class_distribution.values())
        return list(
            np.random.choice(
                labels,
                size=len(X_test),
                p=probabilities))

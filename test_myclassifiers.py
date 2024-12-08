# pylint: skip-file
from mysklearn import myevaluation
from mysklearn.myclassifiers import MyNaiveBayesClassifier
import numpy as np
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyRandomForestClassifier

import pytest
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from collections import Counter

# Test data
X_train = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train = [
    "False",
    "False",
    "True",
    "True",
    "True",
    "False",
    "True",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False"
]

X_test = [
    ["Junior", "Java", "yes", "no"],
    ["Mid", "Python", "no", "no"],
    ["Senior", "R", "yes", "yes"]
]


def test_random_forest_fit():
    """Test the fit method of MyRandomForestClassifier."""
    forest_classifier = MyRandomForestClassifier(n_trees=10, max_features=2)
    forest_classifier.fit(X_train, y_train)

    # Check that trees were created
    assert forest_classifier.trees is not None
    assert len(forest_classifier.trees) > 0

    # Check that the trees are instances of MyDecisionTreeClassifier
    assert all(isinstance(tree, MyDecisionTreeClassifier) for tree in forest_classifier.trees)

    # Check that the number of trees does not exceed n_trees
    assert len(forest_classifier.trees) <= forest_classifier.n_trees


def test_random_forest_predict():
    """Test the predict method of MyRandomForestClassifier."""
    forest_classifier = MyRandomForestClassifier(n_trees=10, max_features=2)
    forest_classifier.fit(X_train, y_train)

    # Predict on the test data
    predictions = forest_classifier.predict(X_test)

    # Check that predictions match the length of the test data
    assert len(predictions) == len(X_test)

    # Check that predictions are among the labels in the training data
    assert all(pred in set(y_train) for pred in predictions)


def test_random_forest_consistency():
    """Test the consistency of MyRandomForestClassifier."""
    forest_classifier = MyRandomForestClassifier(n_trees=10, max_features=2)
    forest_classifier.fit(X_train, y_train)

    # Predict multiple times to ensure consistency
    predictions1 = forest_classifier.predict(X_test)
    predictions2 = forest_classifier.predict(X_test)
    assert predictions1 == predictions2


def test_random_forest_parameterization():
    """Test different parameter settings for Random Forest."""
    for n_trees, max_features in [(5, 2), (10, 3), (20, 4)]:
        forest_classifier = MyRandomForestClassifier(n_trees=n_trees, max_features=max_features)
        forest_classifier.fit(X_train, y_train)

        # Check that the correct number of trees was created
        assert len(forest_classifier.trees) == n_trees

        # Check that predictions are valid
        predictions = forest_classifier.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in set(y_train) for pred in predictions)




def test_decision_tree_classifier_fit():

    interview_tree_solution = ["Attribute", "att0",
                               ["Value", "Junior",
                                ["Attribute", "att3",
                                 ["Value", "no",
                                            ["Leaf", "True", 3, 5]
                                  ],
                                 ["Value", "yes",
                                  ["Leaf", "False", 2, 5]
                                  ]
                                 ]
                                ],
                               ["Value", "Mid",
                                ["Leaf", "True", 4, 14]
                                ],
                               ["Value", "Senior",
                                ["Attribute", "att2",
                                 ["Value", "no",
                                  ["Leaf", "False", 3, 5]
                                  ],
                                 ["Value", "yes",
                                  ["Leaf", "True", 2, 5]
                                  ]
                                 ]
                                ]
                               ]

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)
    assert tree.tree == interview_tree_solution

    iphone_tree_solution = [
        "Attribute", "att0",
        [
            "Value", 1,
            [
                "Attribute", "att1",
                [
                    "Value", 1,
                    ["Leaf", "yes", 1, 5]
                ],
                [
                    "Value", 2,
                    [
                        "Attribute", "att2",
                        [
                            "Value", "excellent",
                            ["Leaf", "yes", 1, 2]
                        ],
                        [
                            "Value", "fair",
                            ["Leaf", "no", 1, 2]
                        ]
                    ]
                ],
                [
                    "Value", 3,
                    ["Leaf", "no", 2, 5]
                ]
            ]
        ],
        [
            "Value", 2,
            [
                "Attribute", "att2",
                [
                    "Value", "excellent",
                    [
                        "Attribute", "att1",
                        [
                            "Value", 1,
                            ["Leaf", "no", 1, 2]
                        ],
                        [
                            "Value", 2,
                            ["Leaf", "yes", 1, 2]
                        ],
                        [
                            "Value", 3,
                            ["Leaf", "no", 2, 4]
                        ]
                    ]
                ],
                [
                    "Value", "fair",
                    ["Leaf", "yes", 6, 10]
                ]
            ]
        ]
    ]
    iphone_tree = MyDecisionTreeClassifier()
    iphone_tree.fit(X_train_iphone, y_train_iphone)
    print(iphone_tree.tree)
    assert iphone_tree.tree == iphone_tree_solution


def test_decision_tree_classifier_predict():
    # True, False, None
    test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"], ["Intern", "Java", "yes", "yes"]]
    expected = ["True", "False", None]
    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)
    pred = tree.predict(test)
    assert pred == expected

    test = [[2, 2, "fair"], [1, 1, "excellent"]]
    expected = ["yes", "yes"]
    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_iphone, y_train_iphone)
    pred = tree.predict(test)
    assert pred == expected


# pylint: skip-file
"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #6
11/7/24
I completed the bonus

Description: This is a data science story about evaluating classifiers using different performance matrixs. This file
holds my tests for my classification
"""


# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6]  # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# MA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
y_train_train = [
    "on time",
    "on time",
    "on time",
    "late",
    "on time",
    "very late",
    "on time",
    "on time",
    "very late",
    "on time",
    "cancelled",
    "on time",
    "late",
    "on time",
    "very late",
    "on time",
    "on time",
    "on time",
    "on time",
    "on time"]


def test_naive_bayes_classifier_fit():

    # In class example (lab task 1)

    in_class_priors = {
        "yes": 5 / 8,  # 5 out of 8 instances are "yes"
        "no": 3 / 8    # 3 out of 8 instances are "no"
    }
    in_class_posteriors = {
        "yes": {
            # probabilities of att1 values given "yes"
            "att1": {1: 4 / 5, 2: 1 / 5},
            # probabilities of att2 values given "yes"
            "att2": {5: 2 / 5, 6: 3 / 5}
        },
        "no": {
            # probabilities of att1 values given "no"
            "att1": {1: 2 / 3, 2: 1 / 3},
            # probabilities of att2 values given "no"
            "att2": {5: 2 / 3, 6: 1 / 3}
        }
    }
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_inclass_example
    naive_class.fit(X_train_inclass_example, y_train_inclass_example)
    assert naive_class.priors == in_class_priors
    assert naive_class.posteriors == in_class_posteriors

    # MA7

    phone_priors = {
        "yes": 10 / 15,  # 10 out of 15 instances are "yes"
        "no": 5 / 15    # 5 out of 15 instances are "no"
    }
    phone_posteriors = {

        "yes": {
            # probabilities of standing values given "yes"
            "standing": {1: 2 / 10, 2: 8 / 10},
            # probabilities of job status values given "yes"
            "job_status": {1: 3 / 10, 2: 4 / 10, 3: 3 / 10},
            # probabilities of credit rating values given "yes"
            "credit_rating": {"fair": 7 / 10, "excellent": 3 / 10}
        },
        "no": {
            # probabilities of standing values given "no"
            "standing": {1: 3 / 5, 2: 2 / 5},
            # probabilities of job status values given "no"
            "job_status": {1: 1 / 5, 2: 2 / 5, 3: 2 / 5},
            # probabilities of credit rating values given "no"
            "credit_rating": {"fair": 2 / 5, "excellent": 3 / 5}
        }
    }
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_iphone
    naive_class.fit(X_train_iphone, y_train_iphone)
    assert naive_class.priors == phone_priors
    assert naive_class.posteriors == phone_posteriors

    # BRAMER

    bramer_priors = {
        "on time": 14 / 20,     # 0.70
        "late": 2 / 20,         # 0.10
        "very late": 3 / 20,    # 0.15
        "cancelled": 1 / 20     # 0.05
    }
    bramer_posteriors = {
        "on time": {
            "day": {
                "weekday": 9 / 14,    # 0.64
                "saturday": 2 / 14,   # 0.14
                "holiday": 2 / 14,     # 0.14
                "sunday": 1 / 14,     # 0.07

            },
            "season": {
                "spring": 4 / 14,     # 0.29
                "winter": 2 / 14,     # 0.21
                "summer": 6 / 14,     # 0.36
                "autumn": 2 / 14,     # 0.14

            },
            "wind": {
                "none": 5 / 14,       # 0.29
                "normal": 5 / 14,      # 0.50
                "high": 4 / 14,       # 0.21

            },
            "rain": {
                "none": 5 / 14,       # 0.36
                "slight": 8 / 14,     # 0.57
                "heavy": 1 / 14       # 0.07
            }
        },
        "late": {
            "day": {
                "weekday": 1 / 2,    # 0.5
                "saturday": 1 / 2,   # 0.5
                "holiday": 0,         # 0
                "sunday": 0,         # 0

            },
            "season": {
                "spring": 0,         # 0
                "winter": 2 / 2,      # 1.0
                "summer": 0,         # 0
                "autumn": 0,         # 0

            },
            "wind": {
                "none": 0,           # 0
                "normal": 1 / 2,     # 0.5
                "high": 1 / 2,       # 0.5
            },
            "rain": {
                "none": 1 / 2,       # 0.5
                "slight": 0,         # 0
                "heavy": 1 / 2       # 0.5
            }
        },
        "very late": {
            "day": {
                "weekday": 3 / 3,        # 1
                "saturday": 0,       # 0
                "holiday": 0,         # 0
                "sunday": 0,         # 0

            },
            "season": {
                "spring": 0,         # 0
                "winter": 2 / 3,     # 0.67
                "summer": 0,         # 0
                "autumn": 1 / 3,     # 0.33

            },
            "wind": {
                "none": 0,           # 0
                "normal": 2 / 3,      # 0.67
                "high": 1 / 3,       # 0.33

            },
            "rain": {
                "none": 1 / 3,       # 0.33
                "slight": 0,         # 0
                "heavy": 2 / 3       # 0.67
            }
        },
        "cancelled": {
            "day": {
                "weekday": 0,        # 0
                "saturday": 1,       # 1
                "holiday": 0,         # 0
                "sunday": 0,         # 0
            },
            "season": {
                "spring": 1,         # 1
                "summer": 0,         # 0
                "autumn": 0,         # 0
                "winter": 0          # 0
            },
            "wind": {
                "none": 0,           # 0
                "normal": 0,          # 0
                "high": 1,           # 1

            },
            "rain": {
                "none": 0,           # 0
                "slight": 0,         # 0
                "heavy": 1           # 1
            }
        }

    }
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_train
    naive_class.fit(X_train_train, y_train_train)
    assert naive_class.priors == bramer_priors
    assert naive_class.posteriors == bramer_posteriors


def test_naive_bayes_classifier_predict():
    # Initialize and train classifier on each dataset before prediction tests

    # In-class example (lab task 1)
    in_class_test = [[1, 5]]
    in_class_pred = ["yes"]
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_inclass_example
    naive_class.fit(X_train_inclass_example,
                    y_train_inclass_example)  # Fit before predict
    class_predictions = naive_class.predict(in_class_test)
    assert in_class_pred == class_predictions

    # MA7 example
    test_MA = [[2, 2, "fair"], [1, 1, "excellent"]]
    pred_MA = ["yes", "no"]
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_iphone
    naive_class.fit(X_train_iphone, y_train_iphone)  # Fit before predict
    predictions_MA = naive_class.predict(test_MA)
    assert pred_MA == predictions_MA

    # Bramer example
    bramer_test = [["weekday", "winter", "high", "heavy"], [
        "weekday", "summer", "high", "heavy"], ["sunday", "summer", "normal", "slight"]]
    # Update to expected output class
    bramer_pred = ["very late", "on time", "on time"]
    naive_class = MyNaiveBayesClassifier()
    naive_class.header = header_train
    naive_class.fit(X_train_train, y_train_train)  # Fit before predict
    predictions_bramer = naive_class.predict(bramer_test)
    assert bramer_pred == predictions_bramer


def test_classification_report():
    # Binary test case
    y_true_binary = ["win", "lose", "win", "lose"]
    y_pred_binary = ["win", "win", "lose", "lose"]
    labels_binary = ["win", "lose"]
    report_binary = myevaluation.classification_report(
        y_true_binary, y_pred_binary, labels_binary, output_dict=True)
    expected_binary = {
        "win": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 2},
        "lose": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 2},
        "macro avg": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 4},
        "weighted avg": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 4}}
    assert report_binary == expected_binary

    # Multi-class test case (coffee acidity)
    y_true_multi = ["low", "medium", "high", "low", "medium", "high"]
    y_pred_multi = ["low", "low", "high", "medium", "medium", "high"]
    labels_multi = ["low", "medium", "high"]
    report_multi = myevaluation.classification_report(
        y_true_multi, y_pred_multi, labels_multi, output_dict=True)
    expected_multi = {
        "low": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 2},
        "medium": {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": 2},
        "high": {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": 2},
        "macro avg": {
            "precision": 0.67,
            "recall": 0.67,
            "f1-score": 0.67,
            "support": 6},
        "weighted avg": {
            "precision": 0.67,
            "recall": 0.67,
            "f1-score": 0.67,
            "support": 6}}
    assert report_multi == expected_multi

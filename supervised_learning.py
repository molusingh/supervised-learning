#!/usr/bin/env python

"""
Author: Anmolbir Mann
Email: amann33@gatech.edu
CS7641 Assignment 1: Supervised Learning
"""

import sys
import warnings
import os

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.neural_network import MLPClassifier # Neural Network Classifier
from sklearn.ensemble import AdaBoostClassifier # AdaBoost Classifier
from sklearn.svm import SVC # SVM Classifier
from sklearn.neighbors import KNeighborsClassifier as KNC # KNN Classifier

from Supervised_Learner import Supervised_Learner

if not sys.warnoptions:
    pass
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

RANDOM_SEED = 1994540112
np.random.seed(RANDOM_SEED) # keep results consistent
OUTPUT_FOLDER = "output"
SCORING = "accuracy"
SCORING_LABEL = "Accuracy"
MAX_ITERATIONS = 2000

# run specified experiment and generate results
def run_experiment(sl, CLF, params, learner_name, learning_path, run=True, clf_args={}):
    if not run:
        return

    param_grid = {}
    clf = CLF(**clf_args)
    print("\n")
    for param, param_range, param_name, path, values in params:
        param_grid[param] = param_range
        if len(param_range) > 1 and values is not None:
            print(f"generating validation curve for {learner_name} and param: {param_name}")
            sl.validation_curve(clf, param, param_range, f"{OUTPUT_FOLDER}/{path}", values, param_name, scoring=SCORING)
            sl.plot_validation_curve(f"{OUTPUT_FOLDER}/{path}", learner_name, ylabel=SCORING_LABEL)

    # grid search
    print(f"running grid search for {learner_name}")
    best_params = sl.grid_search(clf, param_grid, scoring=SCORING)
    print(f"best_params\n{best_params}")
    print(f"generating learning curve for {learner_name}:")
    clf =CLF(**best_params)
    sl.learning_curve(clf, f"{OUTPUT_FOLDER}/{learning_path}", scoring=SCORING)
    best_params_pretty = "\n".join(f"{k}: {v}"  for k, v in best_params.items())

    sl.plot_learning_curve(f"{OUTPUT_FOLDER}/{learning_path}", f"{learner_name}\nWith parameters:\n{best_params_pretty}", ylabel=SCORING_LABEL)
    clf, training_time = sl.train(clf)
    acc_score, f1_score, querying_time = sl.predict(clf, True)
    print(f"\nFinal Results for {learner_name}:\nTraining time: {training_time}\t Querying Time: {querying_time}\tAccuracy: {acc_score}\tF1 Score: {f1_score}\nExperiment completed\n")

# get dataset 1 config
def dataset1():
    data = pd.read_csv("data/eye.csv")
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    sl = Supervised_Learner(x, y, dataset_name="Eye State")

    max_depths = np.arange(1, 20, 1)
    dt_params = [
        ("max_depth", max_depths, "Max Depth", "eye_dt_max_depth", max_depths)
    ]

    nn_heights = np.arange(1, 500, 100)
    nn_widths = np.arange(1, 2, 1)
    nn_hidden_layer_sizes = [np.full(i, j) for i in nn_widths for j in nn_heights]
    nn_learning_rates = np.arange(1, 100, 25) * 0.001
    nn_params = [
        ("hidden_layer_sizes", nn_hidden_layer_sizes, "Nodes Per Layer", "eye_nn_height", nn_heights),
        ('learning_rate_init', nn_learning_rates, "learning_rates", "eye_nn_learning_rate", nn_learning_rates),
        ('max_iter', [MAX_ITERATIONS], "max_iter", "eye_nn_max_iter", None),
        ('early_stopping', [True], "early_stopping", "eye_early_stopping", None)
    ]

    num_learners = np.arange(10, 400, 80)
    boost_params = [("n_estimators", num_learners, "Number of Weak Learners", "eye_boost_n_estimators", num_learners)]

    kernels = np.array(['linear', 'poly', 'rbf'])
    svm_params = [('kernel', kernels, "Kernel (0:Linear 1:Poly, 2:rbf)", "eye_svm_kernel", np.array([0, 1, 2]))]

    num_k = np.arange(1, 10, 1)
    knn_params = [
        ('n_neighbors', num_k, "Number of Neighbors", "eye_knn_k", num_k)
    ]

    return sl, dt_params, nn_params, boost_params, svm_params, knn_params, "eye"

# get dataset2 config
def dataset2():
    data = pd.read_csv("data/diabetes.csv")
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    sl = Supervised_Learner(x, y, dataset_name="Diabetes")

    max_depths = np.arange(1, 24, 3)
    dt_params = [
        ("max_depth", max_depths, "Max Depth", "diabetes_dt_max_depth", max_depths)
    ]

    nn_heights = np.arange(1, 100, 8)
    nn_widths = np.arange(1, 2, 1)
    nn_hidden_layer_sizes = [np.full(i, j) for i in nn_widths for j in nn_heights]
    nn_learning_rates = np.arange(1, 100, 25) * 0.001
    nn_params = [
        ("hidden_layer_sizes", nn_hidden_layer_sizes, "Nodes Per Layer", "diabetes_nn_height", nn_heights),
        ('learning_rate_init', nn_learning_rates, "learning_rates", "diabetes_nn_learning_rate", nn_learning_rates),
        ('max_iter', [MAX_ITERATIONS], "max_iter", "diabetes_nn_max_iter", None),
        ('early_stopping', [True], "early_stopping", "diabetes_early_stopping", None)
    ]

    num_learners = np.arange(5, 200, 20)
    boost_params = [("n_estimators", num_learners, "Number of Weak Learners", "diabetes_boost_n_estimators", num_learners)]

    kernels = np.array(['linear', 'poly', 'rbf'])
    svm_params = [('kernel', kernels, "Kernel (0:Linear 1:Poly, 2:rbf)", "diabetes_svm_kernel", np.array([0, 1, 2]))]

    num_k = np.arange(1, 40, 2)
    knn_params = [
        ('n_neighbors', num_k, "Number of Neighbors", "diabetes_knn_k", num_k),
    ]

    return sl, dt_params, nn_params, boost_params, svm_params, knn_params, "diabetes"

# run specified experiments for given dataset
def execute_dataset(dataset_info, run_dt=True, run_nn=True, run_boost=True, run_svm=True, run_knn=True):
    sl, dt_params, nn_params, boost_params, svm_params, knn_params, label = dataset_info
    run_experiment(sl, DecisionTreeClassifier, dt_params, "Decision Tree", f"{label}_dt", run_dt)
    run_experiment(sl, MLPClassifier, nn_params, "Neural Network", f"{label}_nn", run_nn, {"max_iter": MAX_ITERATIONS, "early_stopping": True})
    run_experiment(sl, AdaBoostClassifier, boost_params, "AdaBoost Classifier Using Decision Stumps", f"{label}_boost", run_boost)
    run_experiment(sl, SVC, svm_params, "SVM Classifier", f"{label}_svm", run_svm)
    run_experiment(sl, KNC, knn_params, "KNN Classifier", f"{label}_knn", run_knn)

def main(args):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if len(args) < 6:
        raise Exception("Not enough arguments")

    use_dataset1, run_dt, run_nn, run_boost, run_svm, run_knn = args
    print(f"Using Dataset {1 if use_dataset1 else 2}")
    print(f"run_dt: {run_dt}\trun_nn: {run_nn}\trun_boost: {run_boost}\nrun_svm: {run_svm}\trun_knn: {run_knn}\n")
    dataset = dataset1() if use_dataset1 else dataset2()
    execute_dataset(dataset, run_dt, run_nn, run_boost, run_svm, run_knn)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = [i != '0' for i in args]
    if len(args) > 6:
        raise Exception("Too many arguments, 6 max")
    args = args + [True for i in range(6 - len(args))]
    main(args)
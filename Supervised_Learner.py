import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import time

from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.neural_network import MLPClassifier # Neural Network Classifier
from sklearn.ensemble import AdaBoostClassifier # AdaBoost Classifier
from sklearn.svm import SVC # SVM Classifier
from sklearn.neighbors import KNeighborsClassifier as KNC # KNN Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score, f1_score #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV, cross_val_score

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# imports for decision tree visualization
from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
import pydotplus

RANDOM_SEED = 1994540101
np.random.seed(RANDOM_SEED) # keep results consistent

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=RANDOM_SEED)


class Supervised_Learner(object):
    def __init__(self,  x, y, ts=0.2, rs=1, dataset_name=""):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts) # 70% training and 30% test
        x_train, y_train = smote.fit_resample(x_train, y_train)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        if len(dataset_name) >= 0:
            self.dataset_name += ":\n"

    def generate_decision_tree_image(self, dtc, output_file="output/decision_tree.png"):
        if dtc.max_depth is None or dtc.max_depth >= 10:
            return None
        dot_data = StringIO()
        export_graphviz(dtc, out_file=dot_data,  filled=True, rounded=True,
                        special_characters=True,feature_names = self.x_train.columns.values,class_names=['0','1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(output_file)
        return Image(graph.create_png())

    def cross_validate(self, learner, cv=5, scoring="f1"):
        name = get_classifier_name(learner)
        learner = Pipeline([('scaler', RobustScaler()), (name, learner)])
        scores = cross_val_score(learner, self.x_train, self.y_train, cv=cv, scoring=scoring)
        return scores.mean()

    def predict(self, learner, test_set=False):
        data = (self.x_test, self.y_test) if test_set else (self.x_train, self.y_train)
        x, y = data
        name = get_classifier_name(learner)
        start_time = time.time()
        y_pred = learner.predict(x)
        training_time = time.time() - start_time
        return accuracy_score(y, y_pred), f1_score(y, y_pred), training_time

    def train(self, learner):
        start_time = time.time()
        name = get_classifier_name(learner)
        pipeline = Pipeline([('scaler', RobustScaler()), (name, learner)])
        pipeline.fit(self.x_train, self.y_train)
        return pipeline, time.time() - start_time

    def learning_curve(self, learner, output_path, scoring="f1"):
        name = get_classifier_name(learner)
        learner = Pipeline([('scaler', RobustScaler()), (name, learner)])
        train_size, train_score, valid_score = learning_curve(learner, self.x_train, self.y_train, scoring=scoring)
        train_score = np.mean(train_score, axis = 1)
        valid_score = np.mean(valid_score, axis = 1)

        results = {"train_size": train_size.tolist(), "train_score": train_score.tolist(), "valid_score": valid_score.tolist()}
        with open(f"{output_path}_learning_curve.json", 'w') as fp:
            json.dump(results, fp)

    def plot_learning_curve(self, datasource, learner_name, show=False, ylabel="F1 Score"):
        with open(f"{datasource}_learning_curve.json", 'r') as fp:
            results = json.load(fp)
        train_size = results["train_size"]
        train_score = results["train_score"]
        valid_score = results["valid_score"]

        # Creating the plot
        plt.plot(train_size, train_score,  label = "Training Score")
        plt.plot(train_size, valid_score, label = "Cross Validation Score")
        plt.title(f"{self.dataset_name}Learning Curve for {learner_name}")
        plt.xlabel("Training Examples")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.legend(loc = 'best')
        plt.savefig(f"{datasource}_learning_curve.png")
        if show:
            plt.show()
        else:
            plt.close()


    def validation_curve(self, learner, param, param_range, output_file, params, param_name, scoring="f1"):
        name = get_classifier_name(learner)
        learner = Pipeline([('scaler', RobustScaler()), (name, learner)])
        param = f"{name}__{param}"
        train_score, valid_score = validation_curve(learner, self.x_train, self.y_train,param_name=param,
                                                    param_range=param_range,scoring=scoring)
        train_score = np.mean(train_score, axis = 1)
        valid_score = np.mean(valid_score, axis = 1)


        results = {"params": params.tolist(), "train_score": train_score.tolist(), "valid_score": valid_score.tolist(), "param_name": param_name}
        with open(f"{output_file}_validation_curve.json", 'w') as fp:
            json.dump(results, fp)

    def plot_validation_curve(self, datasource, title, show=False, ylabel="F1"):
        with open(f"{datasource}_validation_curve.json", 'r') as fp:
            results = json.load(fp)

        params = results["params"]
        train_score = results["train_score"]
        valid_score = results["valid_score"]
        param_name = results["param_name"]

        # Creating the plot
        plt.xticks(params)
        plt.plot(params, train_score,  label = "Training Score")
        plt.plot(params, valid_score, label = "Cross Validation Score")
        plt.title(f"{self.dataset_name}Validation Curve for {title}")
        plt.xlabel(param_name)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.legend(loc = 'best')
        plt.savefig(f"{datasource}_validation_curve.png")
        if show:
            plt.show()
        else:
            plt.close()

    def grid_search(self, learner, param_grid, scoring="f1"):
        name = get_classifier_name(learner)
        learner = Pipeline([('scaler', RobustScaler()), (name, learner)])
        param_grid = {f"{name}__{key}":value for key, value in param_grid.items()}
        grid = GridSearchCV(learner, param_grid, verbose = 1,scoring=scoring)
        grid.fit(self.x_train, self.y_train)
        best_params = grid.best_params_
        best_params = {key.replace(f"{name}__", ""):value for key, value in best_params.items()}
        return best_params

def get_classifier_name(clf):
    if isinstance(clf, DecisionTreeClassifier):
        return "dt"
    elif isinstance(clf, MLPClassifier):
        return "nn"
    elif isinstance(clf, AdaBoostClassifier):
        return "boost"
    elif isinstance(clf, SVC):
        return "svm"
    else:
        return "knn"
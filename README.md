# cs7641-supervised-learning

# Conda Setup Instructions (need conda, can get miniconda off chocolatey for windows or homebrew on mac)
### Using conda to create python environment
conda env create -f environment.yml

### activate the environemnt
conda activate cs7641

### if needed, add debugger
jupyter labextension install @jupyterlab/debugger

### update environment after changes to environment.yml file (deactivate env first)
conda env update --file environment.yml --prune

### Open up jupyter lab to access notebook
jupyter lab

# export notebook to python script if wish to run using python script
jupyter nbconvert --to script supervised_learning_noteebook.ipynb

# recommend running through jupyter lab, but can also run exported python file via below command
ipython supervised_learning_notebook.py

# generate final results for first dataset, outputs charts in ./output directory
python supervised_learning.py 

# generate final results for second dataset, outputs charts in ./output directory
python supervised_learning.py 0

# generate specific results
supervised_learning.py takes 6 arguments below, where value of 0 equals False, anything else is True, unspecified arguments default to 0
use_dataset1, run_dt, run_nn, run_boost, run_svm, run_knn
if use_dataset is True, uses first dataset, else second

References:

Decision Trees
https://www.datacamp.com/community/tutorials/decision-tree-classification-python

Neural Networks
https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

Boosting:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
https://www.tutorialspoint.com/scikit_learn/scikit_learn_boosting_methods.htm
https://www.youtube.com/watch?v=LsK-xG1cLYA

SVM:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

KNN:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

Validation and Learning Curves:
https://www.geeksforgeeks.org/validation-curve/
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html
https://scikit-learn.org/stable/modules/learning_curve.html

Grid Search
https://www.mygreatlearning.com/blog/gridsearchcv/

Fixing Class Imbalances
https://towardsdatascience.com/how-to-effortlessly-handle-class-imbalance-with-python-and-smote-9b715ca8e5a7

Dataset Sources
diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database
Eye Data: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
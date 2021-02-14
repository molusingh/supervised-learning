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

### Open up jupyter lab
jupyter lab

# export notebook to python script if wish to run using python script
jupyter nbconvert --to script supervised_learning.ipynb

# recommend running through jupyter lab, but can also run exported python file via below command
ipython supervised_learning.py

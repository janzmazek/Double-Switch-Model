# Double-Switch-Model
A computational model of pancreatic alpha cell glucagon secretion, implementing metabolic (glucose, FFA) and signalling (cAMP) pathways.

# Prerequisites
Python scientific stack (numpy, scipy, matplotlib) and cython must be pre-installed.

# Setup
Part of the model is written in Cython. Compile the file via the command:
```shell
python src/setup_secretion.py build_ext --inplace
```

# Running the model
Run the model via the command
```shell
python model.py
```
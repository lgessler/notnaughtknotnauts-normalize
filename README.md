# Normalize

This repository contains two systems for orthographic normalization created for a final project in Empirical Methods in Natural Language Processing. A full writeup for this project can be found at `./Report/report.pdf`.

The `Notebooks` folder contains some of the original prototypes of the codebase but these notebooks are considered deprecated. For the final code, please see the `src` folder.

# Usage

Both systems require Python >=3.6. We recommend you use conda to create a fresh
environment to install the dependencies:

```bash
conda create -n normalize python=3.6
conda activate normalize
# for Keras-based system
conda install keras tensorflow-gpu scikit-learn pandas
# for AllenNLP-based system
# (if this doesn't work, you might need to add the --ignore-installed PyYAML flag)
pip install allennlp overrides
```

Running the systems is now simple. Look inside the files for command-line
and other parameters.

```bash
cd src
# LSTM system
python main.py
# transformer system
python anlp.py
```

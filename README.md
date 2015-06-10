# Genomic Selection / Prediction Using Neural Networks

This python code leverages the pybrain and scikit-learn libraries
to perform genomic selection on some genotypic and phenotypic data.
It then compares prediction accuracy between prediction methods.

This code is intended to be run using a python virtualenvironment 
on Linux. Set up a virtualenvironment by running the script below.
The install may take a few minutes depending on the speed of 
your processor and available memory.

```shell
sudo pip install virtualenvwrapper
source $(which virtualenvwrapper)
mkvirtualenv genomic-neuralnet
pip install -r requirements.txt
```

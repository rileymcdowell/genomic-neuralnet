# Genomic Selection / Phenotypic Prediction Using Neural Networks

This python code leverages the pybrain and scikit-learn libraries, 
as well as a built-in artifical neural network implementation
to perform genomic selection on genotypic and phenotypic data.
It compares prediction accuracy between prediction methods.

The purpose of this codebase is to evaluate the 
predictive performance of neural networks and other 
alternate statistical modeling techniques to standard 
mixed linear models that have historically been used for
this purpose.


## Running the code

This is intended to be run using a python virtualenvironment 
on Linux. Set up a virtualenvironment by running the script below.

```shell
sudo pip install virtualenvwrapper
source $(which virtualenvwrapper.sh)
mkvirtualenv genomic-neuralnet
pip install -r requirements.txt
```

To stop working on the code and resume using your 
system's python executable, deactivate the virtualenvironment using the
deactivate command.

```shell
deactivate
```

To continue working on the code, simply say that you wish to work in
the genomic-neuralnet virtualenvironment again.

```shell
workon genomic-neuralnet
```

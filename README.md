# Fintech_Module13

![startup.jpg](https://github.com/nielsdehaan1977/Fintech_Module13/blob/main/Images/startup.jpg)
---
# Neural Networks. 
---
## This jupyter notebook can be used as a template to create a binary classifier model that can to a certain degree predict whether an applicant will become a successful business. The model utilizes TensorFlow library to design a binary classification deep neural network model. This model use a dataset’s that contains information of more than 34,000 startup organizations and tries to predict whether a startup will be successful based on the features in the dataset. The notebook takes into consideration the number of inputs before determining the number of layers that the model will contain or the number of neurons on each layer. Then it compiles and fits the model and evaluates the binary classification model by calculating the model’s loss and accuracy.


![Neural_Networks_2.jpg](https://github.com/nielsdehaan1977/Fintech_Module13/blob/main/Images/Neural_Networks_2.jpg)


## venture_funding_with_deep_learning.ipynb
---
### This notebook can be used as a template to build a model that can be used to predict whether a startup loan applicant will become a succesful business based upon a binary classification model. 
---
The tool can help to predict if a startup will become a succesful business. 
* The tool goes through on the following steps: 
1. Prepare the data for use on a neural network model.
2. Compile and evaluate a binary classification model using a neural network.
3. Optimize the neural network model.
---
## Table of Content

- [Tech](#technologies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Contributor(s)](#contributor(s))
- [License(s)](#license(s))

---
## Tech

This project leverages python 3.9 and Jupyter Lab with the following packages:

* `Python 3.9`
* `Jupyter lab`

* [JupyterLab](https://jupyter.org/) - Jupyter Lab is the latest web-based interactive development environment for notebooks, code, and data.

* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

* [numpy](https://numpy.org/doc/stable/index.html) - NumPy is the fundamental package for scientific computing in Python.

* [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) - The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

* [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - Compute confusion matrix to evaluate the accuracy of a classification.

* [classification_report_imbalanced](https://imbalanced-learn.org/dev/references/generated/imblearn.metrics.classification_report_imbalanced.html) - Build a classification report based on metrics used with imbalanced dataset.

---

## Installation Guide

### Before running the application first install the following dependencies in either Gitbash or Terminal. (If not already installed)

#### Step1: Activate dev environment in Gitbash or Terminal to do so type:
```python
    conda activate dev
```
#### Step2: install the following libraries (if not installed yet) by typing:
```python
    pip install pandas
    pip install --upgrade tensorflow
    
```
#### Step3: Start Jupyter Lab
Jupyter Lab can be started by:
1. Activate your developer environment in Terminal or Git Bash (already done in step 1)
2. Type "jupyter lab --ContentsManager.allow_hidden=True" press enter (This will open Jupyter Lab in a mode where you can also see hidden files)

![JupyterLab](https://github.com/nielsdehaan1977/Fintech_Module12/blob/main/Images/JupyterLab.PNG)


## Usage

To use the credit_risk_resampling jupyter lab notebook, simply clone the full repository and open the **credit_risk_resampling.ipynb** file in Jupyter Lab. 

The tool will go through the following steps:

### Import the Data
* Import of data to analyze

### Prepare the Data
* Split the data into Training and Testing Sets

### Create a Logistics Regression model with original data
* How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

### Create a Logistics Regression model with resampled training data
* How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

### Write a Credit Risk Analysis Report
* **Overview of the analysis:** Explains the purpose of the analysis.
* **Results:** Describes the balanced accuracy scores and the precision and recall scores of both machine learning models.
* **Summary:** Summary of the results from the machine learning models. Compares the two versions of the dataset predictions. Includes recommendations for the model to use, if any, on the original vs. the resampled data.


## Contributor(s)

This project was created by Niels de Haan (nlsdhn@gmail.com)

---

## License(s)

MIT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub watchers](https://img.shields.io/badge/Watchers-1-blue)](https://github.com/Anghille/MachineLearningAlgo/watchers)
[![Pull-Requests Welcome](https://img.shields.io/badge/Pull%20Request-Welcome-blue)](https://github.com/Anghille/MachineLearningAlgo/pulls)

[![python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v1.0.0-blue)](https://github.com/Anghille/MachineLearningAlgo#versioning)

# Introduction

This repository is used to show-case the implementation of most known machine-learning algorithms. From linear regression to SVM, the goal is to develop each model by hand using mostly numpy. Our goal is to be scikit-learn free!

Why did we do that? By implementing each standard algorithms by hand, we had to dive into the core of each algorithm, and understand the maths behind it. Implementing it in a big git project help us develop good practices such as good commits, comments, tests, docstrings, clean and efficient code. We also read a lot the scikit-learn source code, not to copy it, but understand how they had structured the functions, folders and classes, and what was the best way to create tests, data-type checking and such.  
<br><br>

# ToDo
* Complete/Add Tests for utilities function / LinearClass / Metrics
* Add functions to check datatype compatibility inside classes and functions
* Add error/raise to functions (help the devs understand where the error comes from!)
* Continue to add algorithms such as ridge, lasso, classification functions, and such


# Versioning

v1.0.0
* Added *LinearRegression Class* with several computation methods (backpropagation or OLS)
* Added *Regression Metrics* (r2, MSE, MAE, MedAE)
* Added *utility function* train_test_split()

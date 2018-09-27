# Homework 2

![python](https://img.shields.io/badge/python-2.7-blue.svg)
![status](https://img.shields.io/badge/status-in%20progress-yellow.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the second homework for the Multiple Classifiers System's class.

## Description

The goal of this homework is to perform an experiment comparing two different prunning strategies for pool of classifiers applied with three different validation sets. Bagging was chosen for generating pools of 100 Perceptrons, which are combined with hard voting. Metrics are collected for each fold in a 10-fold cross-validation setting. They include accuracy, f-measure, AUC and g-mean. Means and standard deviations of these metrics are calculated in order to analyze the results. Two pairwise diversity measures are also calculated for the prunned ensembles.

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 2.7.15
* [NumPy](http://www.numpy.org/) >= 1.15.1
* [SciPy](https://www.scipy.org/) >= 1.1.0
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.19.2


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the CM1 and JM1 software defect prediction datasets in .arff format from the [Promise repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html) and do not change their names
* Place both .arff files inside the data/ folder

### Reproducing

* Enter into the code/ folder in your local repository
* Run the experiment to produce every ensemble's predictions
```
python main.py
```
* Once finished, generate all results
```
python generate_results.py
```

## Project Structure

    .            
    ├── code                             # Code files
    │   ├── generate_results.py          # generate metric results
    │   ├── main.py                      # generate models predictions
    │   ├── prefit_voting_classifier.py  # voting classifier for prefit base classifiers
    │   └── utils.py                     # utils functions
    ├── data                             # Datasets files
    ├── predictions                      # Models predictions files
    ├── results                          # Metrics files
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
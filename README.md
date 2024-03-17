# AdaBoost for Line-based Classification README

This Python script demonstrates an implementation of the AdaBoost algorithm tailored for a classification task. The goal is to classify points relative to lines generated from a dataset, utilizing AdaBoost to improve the classification accuracy over multiple iterations.

## Features

- **Data Preparation**: Load and randomly split a dataset into training and testing sets.
- **Hypothesis Generation**: Generate line hypotheses from pairs of points to serve as potential classifiers.
- **Point Classification**: Classify points as either above or below a line (hypothesis).
- **AdaBoost Algorithm**: Implement the AdaBoost algorithm to select the best line hypotheses and combine them to form a strong classifier.
- **Prediction & Error Calculation**: Predict classifications for training and testing datasets and calculate empirical and true errors.
- **Multiple Runs Analysis**: Function to perform multiple AdaBoost runs to evaluate the average performance over different splits of the dataset.

## Dependencies

- NumPy: For numerical operations.
- Matplotlib (optional): If you wish to add visualizations of the classifiers and data points (not included in the current script).

## Usage

1. Ensure you have a dataset ready in a text file, with data points and their labels. The script expects the path to this file to be specified as `file_path`.
2. The main parts of the script are wrapped into functions, allowing for modular execution and easy integration into larger projects.
3. To run the script as is, simply execute it in an environment where Python and the required packages are installed. Make sure to adjust `file_path` to point to your dataset.

### Functions Overview

- `load_and_split_data(file_path)`: Loads the data from the specified file path, shuffles, and splits it into training and test sets.
- `generate_line_hypotheses(points)`: Generates potential line classifiers from the provided points.
- `classify_point(line, point)`: Classifies a point based on its position relative to a line.
- `adaboost(X, y, T=8)`: Performs the AdaBoost algorithm to improve classification.
- `predict(models, alphas, X)`: Predicts labels for a set of points based on the trained AdaBoost model.
- `calculate_error(y_true, y_pred)`: Calculates the classification error.
- `multiple_adaboost_runs(file_path, runs=50, T=8)`: Runs the AdaBoost algorithm multiple times to average out performance metrics.

### Outputs

The script prints the empirical error and true error for a single AdaBoost run. Additionally, for multiple runs, it prints the average empirical and true errors for up to `T` iterations of the AdaBoost algorithm.

## Customization

- Adjust `T` (the number of AdaBoost iterations) and `runs` (the number of times the entire AdaBoost process is repeated) according to your dataset size and the computational resources available.
- The script can be extended to include data visualization, allowing for a more intuitive understanding of how the AdaBoost algorithm iteratively improves the classifier's performance.
# AdaBoost Classifier

This project implements the AdaBoost (Adaptive Boosting) algorithm for binary classification using either line or circle classifiers. The implementation allows for experimentation with different configurations and provides visualizations of the learned classifiers.

## Prerequisites

Before running the project, ensure that you have the following dependencies installed:

- Python (version 3.6 or higher)
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Files

- `AdaBoost.py`: The main script that runs the AdaBoost algorithm and performs the experiments.
- `AdaBoostService.py`: A utility class that provides various helper functions used by the AdaBoost algorithm.
- `circle_separator.txt`: The dataset file containing data points and their corresponding labels.

## Usage

1. Clone the repository or download the project files.
2. Place the dataset file (`circle_separator.txt`) in the same directory as the scripts.
3. Configure the experiment parameters in the `AdaBoost.py` script:

```python
# Configuration for the experiment
total_runs = 50
iterations_per_run = 8
classifier_shape = 'circle'  # Choose 'line' or 'circle' classifiers
dataset_path = 'circle_separator.txt'
visualize = True
```

4. Run the `AdaBoost.py` script to execute the experiments:

```bash
python AdaBoost.py
```

## Functionality

- The `AdaBoost` class in `AdaBoost.py` encapsulates the AdaBoost algorithm and performs the experiments:

```python
class AdaBoost:
    def __init__(self):
        self.utility = AdaboostService()

    def execute_adaboost(self, shape_type, num_iterations, visualize, i):
        # ...
        for _ in range(num_iterations):
            # ...
            selected_hypotheses.append(optimal_hypothesis)
            hypothesis_weights.append(alpha)
            # ...
        # ...
        return final_training_error, final_test_error, training_error_log, test_error_log
```

- The `AdaBoostService` class in `AdaBoostService.py` provides utility functions for data processing, hypothesis crafting, error evaluation, and visualization:

```python
class AdaboostService:
    def __init__(self):
        pass

    def fetch_dataset(self, file_path: str) -> list:
        # ...

    def split_data(self, dataset: list) -> tuple:
        # ...

    def set_initial_weights(self, data_points: list) -> dict:
        # ...

    def craft_hypotheses(self, data_points: list):
        # ...

    def evaluate_hypothesis(self, hypothesis, weights, data_points, shape):
        # ...

    def make_prediction(self, hypothesis, point, shape):
        # ...

    def validate_model(self, test_data, top_hypotheses, alphas, shape):
        # ...

    def calculate_model_accuracy(self, predictions):
        # ...
    
    def visualize(self, dataset, classifiers, classifier_type):
        # ...
```

- The script runs multiple experiments based on the configured parameters and aggregates the error metrics.
- If visualization is enabled, the script visualizes the learned classifiers using matplotlib.


## Output

The script outputs the following:
- Average training error and empirical test error for each iteration across all runs:

```
Iteration 1: Avg True Training Error = 0.1234, Avg Empirical Test Error = 0.2345
Iteration 2: Avg True Training Error = 0.0987, Avg Empirical Test Error = 0.1234
...
```

- Overall average training error and test error across all runs:

```
Average Training Error (across all runs): 0.1234
Average Test Error (across all runs): 0.2345
```

- Visualization of the learned classifiers (if enabled).
For example:
![alt text](image.png)

## Customization

- You can modify the experiment parameters in the `AdaBoost.py` script to suit your needs.
- The `AdaBoostService` class can be extended or modified to incorporate additional functionality or customize the behavior of the algorithm.
- The visualization can be enhanced or customized by modifying the `visualize` method in the `AdaBoostService` class.

Feel free to explore and experiment with different configurations and datasets to gain insights into the AdaBoost algorithm's performance and behavior.
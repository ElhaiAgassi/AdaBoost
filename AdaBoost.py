import numpy as np
import logging

# Setup basic logging for monitoring the progress
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

file_path = "circle_separator.txt"  # Path to the dataset file


def load_and_split_data(file_path):
    """
    Loads data from the specified file path, shuffles it,
    and splits it evenly into training and testing datasets.
    """
    data = np.loadtxt(file_path)  # Load data from text file
    np.random.shuffle(data)  # Shuffle data to ensure random distribution
    split_index = len(data) // 2  # Find the mid-point for splitting
    # Split data into training and testing sets
    train_data, test_data = data[:split_index, :], data[split_index:, :]
    return train_data, test_data


def generate_hypotheses(points, classifier_type, sample_size=10):
    """
    Generates a list of hypotheses (lines or circles) based on the input points.
    The function can selectively generate lines, circles, or both, depending on
    the classifier_type argument. It also samples a subset of points to reduce
    computational load.

    Args:
    - points: Array of points to be used for generating hypotheses.
    - classifier_type: A string indicating the type of classifier ('line', 'circle', or 'both').
    - sample_size: The number of points to sample for hypothesis generation.

    Returns:
    - A list of tuples where each tuple contains a classifier type and its parameters.
    """
    # Sample points to reduce the number of generated hypotheses
    points = points[np.random.choice(
        points.shape[0], sample_size, replace=False)]
    hypotheses = []
    # Generate line hypotheses
    if classifier_type in ['line', 'both']:
        logging.info("Generating line hypotheses")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # Calculate coefficients for the line equation
                a = y2 - y1
                b = x1 - x2
                c = x2*y1 - x1*y2
                hypotheses.append(('line', (a, b, c)))
    # Generate circle hypotheses
    if classifier_type in ['circle', 'both']:
        logging.info("Generating circle hypotheses")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                center = points[i]
                radius_point = points[j]
                # Calculate the radius of the circle
                radius = np.linalg.norm(center - radius_point)
                hypotheses.append(('circle', (center, radius)))
    return hypotheses


def classify_point(hypothesis, point):
    """
    Classifies a point as inside or outside (1 or -1) based on a given hypothesis.

    Args:
    - hypothesis: A tuple containing the hypothesis type ('line' or 'circle') and its parameters.
    - point: The point to classify.

    Returns:
    - Classification result (1 or -1).
    """
    type, model = hypothesis
    if type == 'line':
        a, b, c = model
        # Classify based on the line equation
        return 1 if a*point[0] + b*point[1] + c > 0 else -1
    elif type == 'circle':
        center, radius = model
        # Classify based on distance to the circle's center
        distance = np.linalg.norm(point[:2] - center)
        return 1 if distance <= radius else -1


def adaboost(X, y, T=8, classifier_type='line'):
    """
    Performs the AdaBoost algorithm to boost weak classifiers (line or circle).

    Args:
    - X: Feature set (points).
    - y: Labels for each point.
    - T: Number of boosting rounds.
    - classifier_type: Type of weak classifier ('line', 'circle', or 'both').

    Returns:
    - A list of models (hypotheses) and their corresponding weights (alphas).
    """
    logging.info("Starting AdaBoost with classifier type: %s", classifier_type)
    n_samples = len(X)
    # Initialize weights evenly across all samples
    weights = np.full(n_samples, 1/n_samples)
    models = []
    alphas = []

    for t in range(T):
        logging.info("AdaBoost iteration: %d", t + 1)
        hypotheses = generate_hypotheses(X, classifier_type)
        errors = np.ones(len(hypotheses))

        # Evaluate and select the best hypothesis based on weighted error
        for i, hypothesis in enumerate(hypotheses):
            predictions = np.array(
                [classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])
        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        best_error = errors[best_hypothesis_idx]
        # Compute weight (alpha) for the selected hypothesis
        alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        models.append(best_hypothesis)
        alphas.append(alpha)

        # Update sample weights for the next iteration
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

    return models, alphas


def classify_point(hypothesis, point):
    """
    Classifies a point based on a given hypothesis. The hypothesis can be either a line or a circle.

    Parameters:
    - hypothesis: A tuple containing the hypothesis type ('line' or 'circle') and its parameters.
    - point: The (x, y) coordinates of the point to classify.

    Returns:
    - Classification result: 1 for a positive classification, -1 for a negative classification.
    """
    type, model = hypothesis
    if type == 'line':
        # Line classification based on the line equation ax + by + c
        a, b, c = model
        return 1 if a*point[0] + b*point[1] + c > 0 else -1
    elif type == 'circle':
        # Circle classification based on whether the point is inside or outside the circle
        center, radius = model
        distance = np.sqrt((point[0] - center[0]) **
                           2 + (point[1] - center[1])**2)
        return 1 if distance <= radius else -1


def adaboost(X, y, T=8, classifier_type='line'):
    """
    Performs the AdaBoost algorithm to combine weak classifiers into a stronger classifier.

    Parameters:
    - X: Feature set.
    - y: Labels.
    - T: Number of boosting rounds.
    - classifier_type: Type of classifiers to use ('line', 'circle', or 'both').

    Returns:
    - models: The selected models (hypotheses) in each round.
    - alphas: The weight of each selected model.
    """
    n_samples = len(X)
    weights = np.full(n_samples, 1/n_samples)  # Initialize weights evenly
    models = []
    alphas = []

    # Iteratively build the AdaBoost model
    for t in range(T):
        hypotheses = generate_hypotheses(X, classifier_type)
        errors = np.ones(len(hypotheses))

        # Evaluate and select the best hypothesis based on weighted error
        for i, hypothesis in enumerate(hypotheses):
            predictions = np.array(
                [classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])

        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        best_error = errors[best_hypothesis_idx]

        # Compute the model weight (alpha) from the error rate
        alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
        models.append(best_hypothesis)
        alphas.append(alpha)

        # Update the data weights for the next iteration
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

    return models, alphas


def predict(models, alphas, X):
    """
    Makes predictions with the AdaBoost model.

    Parameters:
    - models: The models selected by AdaBoost.
    - alphas: The weights of each model.
    - X: The feature set for prediction.

    Returns:
    - The sign of the aggregated predictions from all models, indicating the class.
    """
    n_samples = len(X)
    final_prediction = np.zeros(n_samples)

    # Aggregate predictions from all models, weighted by their alpha values
    for model, alpha in zip(models, alphas):
        predictions = np.array([classify_point(model, point) for point in X])
        final_prediction += alpha * predictions

    return np.sign(final_prediction)


def calculate_error(y_true, y_pred):
    """
    Calculates the error rate between true labels and predicted labels.

    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    Returns:
    - The error rate as a float.
    """
    return np.mean(y_true != y_pred)


def multiple_adaboost_runs(file_path, runs=50, T=8, classifier_type='line'):
    """
    Executes multiple runs of the AdaBoost algorithm to average out errors.

    Parameters:
    - file_path: Path to the dataset.
    - runs: Number of runs.
    - T: Number of boosting rounds in each run.
    - classifier_type: Type of classifiers to use ('line', 'circle', or 'both').

    Returns:
    - average_empirical_errors: The average training error across all runs.
    - average_true_errors: The average test error across all runs.
    """
    all_empirical_errors = np.zeros((runs, T))
    all_true_errors = np.zeros((runs, T))

    # Perform multiple runs of AdaBoost
    for run in range(runs):
        train_data, test_data = load_and_split_data(file_path)
        X_train = train_data[:, :2]
        y_train = train_data[:, 2]
        X_test = test_data[:, :2]
        y_test = test_data[:, 2]

        models, alphas = adaboost(X_train, y_train, T, classifier_type)

        # Calculate errors for each round of boosting
        for k in range(1, T+1):
            y_train_pred = predict(models[:k], alphas[:k], X_train)
            y_test_pred = predict(models[:k], alphas[:k], X_test)

            empirical_error = calculate_error(y_train, y_train_pred)
            true_error = calculate_error(y_test, y_test_pred)

            all_empirical_errors[run, k-1] = empirical_error
            all_true_errors[run, k-1] = true_error

    # Average the errors across all runs
    average_empirical_errors = np.mean(all_empirical_errors, axis=0)
    average_true_errors = np.mean(all_true_errors, axis=0)

    return average_empirical_errors, average_true_errors


# Example usage
classifier_type = 'circle'  # Option to change to 'line', 'circle', or 'both'
average_empirical_errors, average_true_errors = multiple_adaboost_runs(
    file_path, classifier_type=classifier_type)

for k in range(8):
    print(
        f"k={k+1}, Classifier Type: {classifier_type}: Average Empirical Error = {average_empirical_errors[k]}, Average True Error = {average_true_errors[k]}")

# Demonstrate loading data, training AdaBoost, and making predictions
train_data, test_data = load_and_split_data(file_path)
X_train, y_train = train_data[:, :2], train_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

# Train the AdaBoost model
models, alphas = adaboost(X_train, y_train, T=8,
                          classifier_type=classifier_type)
logging.info("AdaBoost training completed.")

# Make predictions on the test set
y_test_pred = predict(models, alphas, X_test)

# Calculate and print the test error
test_error = calculate_error(y_test, y_test_pred)
print(f"Test Error for '{classifier_type}' classifier: {test_error}")

logging.info("AdaBoost example run completed.")

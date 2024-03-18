import matplotlib.pyplot as plt
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


def generate_hypotheses(points, classifier_type):
    """
    Generates a list of hypotheses (lines or circles) based on the input points.
    The function can selectively generate lines or circles depending on
    the classifier_type argument.

    Args:
    - points: Array of points to be used for generating hypotheses.
    - classifier_type: A string indicating the type of classifier ('line' or 'circle').

    Returns:
    - A list of tuples where each tuple contains a classifier type and its parameters.
    """
    hypotheses = []
    # Generate line hypotheses
    if classifier_type == 'line':
        # logging.info("Generating line hypotheses")
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
    if classifier_type == 'circle':
        # logging.info("Generating circle hypotheses")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                center = points[i]
                radius_point = points[j]
                # Calculate the radius of the circle
                radius = np.linalg.norm(center - radius_point)
                hypotheses.append(('circle', (center, radius)))
    return hypotheses


def adaboost(X, y, R=8, classifier_type='line'):
    """
    Performs the AdaBoost algorithm to boost weak classifiers (line or circle).

    Args:
    - X: Feature set (points).
    - y: Labels for each point.
    - R: Number of boosting rounds.
    - classifier_type: Type of weak classifier ('line', 'circle').

    Returns:
    - A list of models (hypotheses) and their corresponding weights (alphas).
    """
    logging.info("Starting AdaBoost with classifier type: %s", classifier_type)
    n_samples = len(X)
    # Initialize weights evenly across all samples
    weights = np.full(n_samples, 1/n_samples)
    models = []
    alphas = []

    for r in range(R):
        logging.info("AdaBoost iteration: %d", r + 1)
        hypotheses = generate_hypotheses(X, classifier_type)
        errors = np.zeros(len(hypotheses))

        # Evaluate and select the best hypothesis based on weighted error
        for i, hypothesis in enumerate(hypotheses):
            predictions = np.array(
                [classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])

        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        models.append(best_hypothesis)
        best_error = errors[best_hypothesis_idx]

        # Compute weight (alpha) for the selected hypothesis
        alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        alphas.append(alpha)
        best_predictions = np.array(
            [classify_point(best_hypothesis, X[j]) for j in range(n_samples)])

        # Update sample weights for the next iteration
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights)

    return models, alphas


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


def multiple_adaboost_runs(file_path, runs=50, R=8, classifier_type='line'):
    """
    Executes multiple runs of the AdaBoost algorithm to average out errors.

    Parameters:
    - file_path: Path to the dataset.
    - runs: Number of runs.
    - R: Number of boosting rounds in each run.
    - classifier_type: Type of classifiers to use ('line', 'circle', or 'both').

    Returns:
    - average_empirical_errors: The average training error across all runs.
    - average_true_errors: The average test error across all runs.
    """
    all_empirical_errors = np.zeros((runs, R))
    all_true_errors = np.zeros((runs, R))

    # Perform multiple runs of AdaBoost
    for run in range(runs):
        train_data, test_data = load_and_split_data(file_path)
        X_train = train_data[:, :2]
        y_train = train_data[:, 2]
        X_test = test_data[:, :2]
        y_test = test_data[:, 2]

        models, alphas = adaboost(X_train, y_train, R, classifier_type)

        # Calculate errors for each round of boosting
        for k in range(1, R+1):
            y_train_pred = predict(models[:k], alphas[:k], X_train)
            y_test_pred = predict(models[:k], alphas[:k], X_test)

            empirical_error = calculate_error(y_train, y_train_pred)
            true_error = calculate_error(y_test, y_test_pred)

            all_empirical_errors[run, k-1] = empirical_error
            all_true_errors[run, k-1] = true_error

    # Average the errors across all runs
    average_empirical_errors = np.mean(all_empirical_errors, axis=0)
    average_true_errors = np.mean(all_true_errors, axis=0)

    return average_empirical_errors, average_true_errors, models, alphas


# Example usage

# Assuming the rest of the functions (load_and_split_data, generate_hypotheses, adaboost, classify_point, predict, calculate_error) are defined above.


def visualize_adaboost_results(S, models, classifier_type):
    # Plot all points
    for point in S:
        plt.scatter(point[0], point[1],
                    color='red' if point[2] == 1 else 'blue')

    # Set the limits of the plot
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    if classifier_type == 'circle':
        # Plot each circle
        for model in models:
            type, (center, radius) = model
            if type == 'circle':
                circle = plt.Circle(center, radius, color='r', fill=False)
                plt.gca().add_artist(circle)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('AdaBoost Classification')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Setup basic logging for monitoring the progress
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

file_path = "circle_separator.txt"  # Path to the dataset file
classifier_type = 'circle'  # Option to change to 'line', 'circle'
average_empirical_errors, average_true_errors, models, alphas = multiple_adaboost_runs(
    file_path, runs=50, R=8, classifier_type=classifier_type)

# Print average errors for insight
for k in range(8):
    print(
        f"k={k+1}, Classifier Type: {classifier_type}, Average Empirical Error = {average_empirical_errors[k]}, Average True Error = {average_true_errors[k]}")

# Visualize the results using the models and alphas from the last AdaBoost run
# Get the last train data used for visualization
train_data, _ = load_and_split_data(file_path)
visualize_adaboost_results(train_data, models, classifier_type)

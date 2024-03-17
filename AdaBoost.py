import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

file_path = "circle_separator.txt"


def load_and_split_data(file_path):
    data = np.loadtxt(file_path)
    np.random.shuffle(data)
    split_index = len(data) // 2
    train_data, test_data = data[:split_index, :], data[split_index:, :]
    return train_data, test_data


def generate_hypotheses(points, classifier_type, sample_size=10):
    points = points[np.random.choice(
        points.shape[0], sample_size, replace=False)]  # Sampling points
    hypotheses = []
    if classifier_type == 'line' or classifier_type == 'both':
        logging.info("Generating line hypotheses")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                a = y2 - y1
                b = x1 - x2
                c = x2*y1 - x1*y2
                hypotheses.append(('line', (a, b, c)))
    if classifier_type == 'circle' or classifier_type == 'both':
        logging.info("Generating circle hypotheses")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                center = points[i]
                radius_point = points[j]
                radius = np.linalg.norm(center - radius_point)
                hypotheses.append(('circle', (center, radius)))
    return hypotheses


def adaboost(X, y, T=8, classifier_type='line'):
    logging.info("Starting AdaBoost with classifier type: %s", classifier_type)
    n_samples = len(X)
    weights = np.full(n_samples, 1/n_samples)
    models = []
    alphas = []

    for t in range(T):
        logging.info("AdaBoost iteration: %d", t + 1)
        hypotheses = generate_hypotheses(X, classifier_type)
        errors = np.ones(len(hypotheses))

        for i, hypothesis in enumerate(hypotheses):
            predictions = np.array(
                [classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])
        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        models.append(best_hypothesis)
        best_error = errors[best_hypothesis_idx]
        alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
        alphas.append(alpha)

        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

    return models, alphas


def classify_point(hypothesis, point):
    type, model = hypothesis
    if type == 'line':
        a, b, c = model
        return 1 if a*point[0] + b*point[1] + c > 0 else -1
    elif type == 'circle':
        center, radius = model
        distance = np.sqrt((point[0] - center[0]) **
                           2 + (point[1] - center[1])**2)
        return 1 if distance <= radius else -1


def adaboost(X, y, T=8, classifier_type='line'):
    n_samples = len(X)
    weights = np.full(n_samples, 1/n_samples)
    models = []
    alphas = []

    for t in range(T):
        hypotheses = generate_hypotheses(X, classifier_type)
        errors = np.ones(len(hypotheses))
        for i, hypothesis in enumerate(hypotheses):
            predictions = np.array(
                [classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])
        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        best_error = errors[best_hypothesis_idx]
        alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
        models.append(best_hypothesis)
        alphas.append(alpha)
        best_predictions = np.array(
            [classify_point(best_hypothesis, X[j]) for j in range(n_samples)])
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights)

    return models, alphas


def predict(models, alphas, X):
    n_samples = len(X)
    final_prediction = np.zeros(n_samples)
    for model, alpha in zip(models, alphas):
        predictions = np.array([classify_point(model, point) for point in X])
        final_prediction += alpha * predictions
    return np.sign(final_prediction)


def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multiple_adaboost_runs(file_path, runs=50, T=8, classifier_type='line'):
    all_empirical_errors = np.zeros((runs, T))
    all_true_errors = np.zeros((runs, T))

    for run in range(runs):
        train_data, test_data = load_and_split_data(file_path)
        X_train = train_data[:, :2]
        y_train = train_data[:, 2]
        X_test = test_data[:, :2]
        y_test = test_data[:, 2]

        models, alphas = adaboost(X_train, y_train, T, classifier_type)

        for k in range(1, T+1):
            y_train_pred = predict(models[:k], alphas[:k], X_train)
            y_test_pred = predict(models[:k], alphas[:k], X_test)

            empirical_error = calculate_error(y_train, y_train_pred)
            true_error = calculate_error(y_test, y_test_pred)

            all_empirical_errors[run, k-1] = empirical_error
            all_true_errors[run, k-1] = true_error

    average_empirical_errors = np.mean(all_empirical_errors, axis=0)
    average_true_errors = np.mean(all_true_errors, axis=0)

    return average_empirical_errors, average_true_errors


# Example usage
classifier_type = 'circle'  # Change to 'line', 'circle', or 'both'
average_empirical_errors, average_true_errors = multiple_adaboost_runs(
    file_path, classifier_type=classifier_type)

for k in range(8):
    print(
        f"k={k+1}, Classifier Type: {classifier_type}: Average Empirical Error = {average_empirical_errors[k]}, Average True Error = {average_true_errors[k]}")

# Example: Running a single AdaBoost run with logging
train_data, test_data = load_and_split_data(file_path)
X_train = train_data[:, :2]
y_train = train_data[:, 2]
# Change classifier_type as needed
models, alphas = adaboost(X_train, y_train, T=8, classifier_type='line')
logging.info("AdaBoost completed")

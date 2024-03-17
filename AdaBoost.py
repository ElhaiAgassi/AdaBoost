import numpy as np
import matplotlib.pyplot as plt

file_path = "circle_separator.txt"  # Ensure this matches your file path

def load_and_split_data(file_path):
    data = np.loadtxt(file_path)
    np.random.shuffle(data)
    split_index = len(data) // 2
    train_data, test_data = data[:split_index, :], data[split_index:, :]
    return train_data, test_data

def generate_hypotheses(points, classifier_type):
    hypotheses = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if classifier_type in ['line', 'both']:
                # Generate line
                x1, y1 = points[i]
                x2, y2 = points[j]
                a = y2 - y1
                b = x1 - x2
                c = x2*y1 - x1*y2
                hypotheses.append(('line', (a, b, c)))
            if classifier_type in ['circle', 'both']:
                # Generate circle
                center = points[i]
                radius_point = points[j]
                radius = np.sqrt((center[0] - radius_point[0])**2 + (center[1] - radius_point[1])**2)
                hypotheses.append(('circle', (center, radius)))
    return hypotheses

def classify_point(hypothesis, point):
    type, model = hypothesis
    if type == 'line':
        a, b, c = model
        return 1 if a*point[0] + b*point[1] + c > 0 else -1
    elif type == 'circle':
        center, radius = model
        distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
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
            predictions = np.array([classify_point(hypothesis, X[j]) for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])
        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = hypotheses[best_hypothesis_idx]
        best_error = errors[best_hypothesis_idx]
        alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
        models.append(best_hypothesis)
        alphas.append(alpha)
        best_predictions = np.array([classify_point(best_hypothesis, X[j]) for j in range(n_samples)])
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
average_empirical_errors, average_true_errors = multiple_adaboost_runs(file_path, classifier_type=classifier_type)

for k in range(8):
    print(f"k={k+1}, Classifier Type: {classifier_type}: Average Empirical Error = {average_empirical_errors[k]}, Average True Error = {average_true_errors[k]}")

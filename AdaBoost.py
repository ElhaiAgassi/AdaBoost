import numpy as np
import matplotlib.pyplot as plt

file_path = "C:/Users/elhaia/Downloads/circle_separator.txt"

# Corrected function to load and split data


def load_and_split_data(file_path):
    data = np.loadtxt(file_path)
    np.random.shuffle(data)  # Shuffle to ensure random distribution
    split_index = len(data) // 2
    train_data, test_data = data[:split_index, :], data[split_index:, :]
    return train_data, test_data

# Function to generate line hypotheses from corrected pairs of points


def generate_line_hypotheses(points):
    lines = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            lines.append((a, b, c))
    return lines

# Function to classify a point relative to a line


def classify_point(line, point):
    a, b, c = line
    x, y = point[:2]
    return 1 if a*x + b*y + c > 0 else -1

# Corrected AdaBoost algorithm


def adaboost(X, y, T=8):
    n_samples = len(X)
    weights = np.full(n_samples, 1/n_samples)
    models = []
    alphas = []

    for t in range(T):
        line_hypotheses = generate_line_hypotheses(X)
        errors = np.ones(len(line_hypotheses))
        for i, line in enumerate(line_hypotheses):
            predictions = np.array([classify_point(line, X[j])
                                   for j in range(n_samples)])
            errors[i] = np.sum(weights[y != predictions])
        best_hypothesis_idx = np.argmin(errors)
        best_hypothesis = line_hypotheses[best_hypothesis_idx]
        best_error = errors[best_hypothesis_idx]
        alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
        models.append(best_hypothesis)
        alphas.append(alpha)
        best_predictions = np.array(
            [classify_point(best_hypothesis, X[j]) for j in range(n_samples)])
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights)

    return models, alphas

# Function to predict labels based on AdaBoost models


def predict(models, alphas, X):
    n_samples = X.shape[0]
    final_prediction = np.zeros(n_samples)
    for model, alpha in zip(models, alphas):
        predictions = np.array([classify_point(model, point) for point in X])
        final_prediction += alpha * predictions
    return np.sign(final_prediction)

# Function to calculate error


def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


# Load and split the dataset
train_data, test_data = load_and_split_data(file_path)

# Preparing training and test data
X_train = train_data[:, :2]  # Feature set for training
y_train = train_data[:, 2]   # Labels for training
X_test = test_data[:, :2]    # Feature set for testing
y_test = test_data[:, 2]     # Labels for testing

# Running AdaBoost
models, alphas = adaboost(X_train, y_train, T=8)

# Making predictions
y_train_pred = predict(models, alphas, X_train)
y_test_pred = predict(models, alphas, X_test)

# Calculating errors
empirical_error = calculate_error(y_train, y_train_pred)
true_error = calculate_error(y_test, y_test_pred)

print(f"Empirical Error: {empirical_error}")
print(f"True Error: {true_error}")

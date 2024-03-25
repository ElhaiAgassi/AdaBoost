import numpy as np
import matplotlib.pyplot as plt

class AdaBoostService:
    @staticmethod
    def generate_hypotheses(X, classifier_type):
        """Generate a list of hypotheses based on the classifier type (line or circle),
        incorporating line hypothesis generation directly within this method."""
        hypotheses = []
        if classifier_type == 'line':
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    # Directly generate line hypothesis here
                    point1, point2 = X[i], X[j]
                    a = point2[1] - point1[1]
                    b = point1[0] - point2[0]
                    c = point2[0] * point1[1] - point1[0] * point2[1]
                    hypotheses.append(('line', (a, b, c)))
        elif classifier_type == 'circle':
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    center = X[i]
                    radius_point = X[j]
                    radius = np.linalg.norm(center - radius_point)
                    hypotheses.append(('circle', (center, radius)))
        return hypotheses
    
    @staticmethod
    def evaluate_hypothesis(hypothesis, X, y, weights, classifier_type):
        error = 0
        predictions = np.zeros_like(y)
        if classifier_type == 'line':
            a, b, c = hypothesis[1]
            predictions = np.sign(a * X[:, 0] + b * X[:, 1] + c)
        elif classifier_type == 'circle':
            center, radius = hypothesis[1]
            distances = np.sqrt((X[:, 0] - center[0])**2 + (X[:, 1] - center[1])**2)
            predictions = np.sign(distances - radius)
        error = np.sum(weights[y != predictions])
        return error, predictions

    @staticmethod
    def update_weights(weights, alpha, y, predictions):
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)  # Normalize
        return weights

    @staticmethod
    def predict(models, alphas, X, classifier_type):
        final_prediction = np.zeros(len(X))
        for model, alpha in zip(models, alphas):
            if classifier_type == 'line':
                a, b, c = model[1]
                predictions = np.sign(a * X[:, 0] + b * X[:, 1] + c)
            elif classifier_type == 'circle':
                center, radius = model[1]
                distances = np.sqrt((X[:, 0] - center[0])**2 + (X[:, 1] - center[1])**2)
                predictions = np.sign(distances - radius)
            final_prediction += alpha * predictions
        return np.sign(final_prediction)
    
    @staticmethod
    def calculate_error(y_true, y_pred):
            """Calculate the mean error rate between true labels and predicted labels."""
            return np.mean(y_true != y_pred)
    
    @staticmethod
    def load_and_split_data(file_path):
        data = np.loadtxt(file_path, delimiter=None)
        np.random.shuffle(data)
        split_index = len(data) // 2
        train_data, test_data = data[:split_index, :], data[split_index:, :]
        return train_data, test_data
    
    @staticmethod
    def visualize_best_hypotheses(X, y, models, classifier_type):
        """Visualize the training data and overlay the best hypotheses."""
        plt.figure(figsize=(8, 8))
        # Plot data points, color-coded by their labels
        for label, marker in zip([-1, 1], ['rx', 'bo']):
            plt.scatter(X[y == label, 0], X[y == label, 1], marker=marker[1], color=marker[0], label=f"Class {label}")

        # Define x range for plotting line equations
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), num=400)

        if classifier_type == 'line':
            for model in models:
                a, b, c = model[1]
                # Calculate y values based on the line equation ax + by + c = 0
                y_range = (-a * x_range - c) / b
                plt.plot(x_range, y_range, label=f"Line: {a:.2f}x + {b:.2f}y + {c:.2f} = 0")
        elif classifier_type == 'circle':
            for model in models:
                center, radius = model[1]
                circle = plt.Circle(center, radius, color='g', fill=False, linestyle='--', label=f"Circle with radius {radius:.2f}")
                plt.gca().add_artist(circle)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Best Hypotheses Visualization")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
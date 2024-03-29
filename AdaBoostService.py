from matplotlib import pyplot as plt
import numpy as np
from itertools import product
import random

# Re-defining the AdaBoostService with necessary imports
class AdaBoostService:
    """A utility class providing supporting functions for the AdaBoost classifier."""

    def load_data(self, file_path: str) -> list:
        dataset = []
        with open(file_path, 'r') as data_file:
            for line in data_file:
                x, y, label = line.strip().split()
                x, y = float(x), float(y)
                dataset.append((x, y, int(label)))
        return dataset

    def split_dataset(self, dataset: list) -> tuple:
        """Randomly splits the dataset into two subsets."""
        random.shuffle(dataset)
        split_point = int(len(dataset) * 0.5)
        return dataset[:split_point], dataset[split_point:]

    def initialize_weights(self, dataset: list) -> dict:
        """Initializes and returns weights for each item in the dataset."""
        initial_weight = 1.0 / len(dataset)
        return {data: initial_weight for data in dataset}

    def generate_classifiers(self, dataset: list, classifier_type: str) -> list:
        """Generates and returns possible classifiers based on data combinations."""
        if classifier_type == 'line':
            return list(product(dataset, repeat=2))
        elif classifier_type == 'circle':
            return [(center, point) for center in dataset for point in dataset if center != point]

    def evaluate_classifier(self, classifier, weights, dataset, classifier_type):
        error, predictions = 0, {}
        for data in dataset:
            prediction = self.predict(classifier, data, classifier_type)
            real_label = data[2]
            predictions[data] = prediction
            
            if prediction != real_label:
                error += weights[data]
        return error, predictions

    def predict(self, classifier, data, classifier_type):
        """Predicts the label of a given data point using the specified classifier."""
        if classifier_type == 'line':
            point1, point2 = classifier
            determinant = (point2[0] - point1[0]) * (data[1] - point1[1]) - (point2[1] - point1[1]) * (data[0] - point1[0])
            return 1 if determinant > 0 else -1
        elif classifier_type == 'circle':
            center, radius_point = classifier
            radius = ((radius_point[0] - center[0])**2 + (radius_point[1] - center[1])**2)**0.5
            distance = ((data[0] - center[0])**2 + (data[1] - center[1])**2)**0.5
            return 1 if distance <= radius else -1

    def aggregate_predictions(self, test_data, classifiers, alphas, classifier_type):
        """Aggregates predictions from all classifiers for the test data."""
        final_predictions = []
        for data in test_data:
            weighted_sum = sum(alpha * self.predict(classifier, data, classifier_type) for classifier, alpha in zip(classifiers, alphas))
            final_label = 1 if weighted_sum > 0 else -1
            final_predictions.append((data, final_label))
        return final_predictions

    def calculate_accuracy(self, predictions):
        """Calculates the accuracy of predictions."""
        correct = sum(1 for data, predicted in predictions if data[2] == predicted)
        return correct / len(predictions)

    def evaluate_performance(self, test_data, training_data, classifiers, alphas, classifier_type):
        training_predictions = self.aggregate_predictions(training_data, classifiers, alphas, classifier_type)
        test_predictions = self.aggregate_predictions(test_data, classifiers, alphas, classifier_type)

        empirical_accuracy = self.calculate_accuracy(training_predictions)
        true_accuracy = self.calculate_accuracy(test_predictions)

        return 1 - empirical_accuracy, 1 - true_accuracy

    def visualize(self, dataset, classifiers, classifier_type):
        # Convert dataset to a NumPy array for easier indexing
        dataset_np = np.array(dataset)

        # First, plot the dataset points
        for point in dataset_np:
            plt.scatter(point[0], point[1], color='red' if point[2] == 1 else 'blue')

        # Now plot the classifiers based on type
        if classifier_type == 'line':
            for classifier in classifiers:
                point1, point2 = classifier
                # Handle the case for vertical lines to avoid division by zero
                if point1[0] == point2[0]:
                    plt.axvline(x=point1[0], color='green')
                else:
                    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                    intercept = point1[1] - slope * point1[0]
                    x_values = np.linspace(dataset_np[:,0].min(), dataset_np[:,0].max(), 100)
                    y_values = slope * x_values + intercept
                    plt.plot(x_values, y_values, '-g')
        elif classifier_type == 'circle':
            for classifier in classifiers:
                center, point_on_circle = classifier
                radius = np.sqrt((point_on_circle[0] - center[0])**2 + (point_on_circle[1] - center[1])**2)
                circle = plt.Circle(center, radius, color='green', fill=False)
                plt.gca().add_artist(circle)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f'AdaBoost Classification with {classifier_type}')
        plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
        plt.show()
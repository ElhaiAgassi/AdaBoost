import numpy as np
import random
from itertools import combinations
from matplotlib import pyplot as plt

class AdaBoostService:
    """A utility class providing supporting functions for the AdaBoost classifier."""
    
    def __init__(self):
        pass

    def load_data(self, file_path: str) -> list:
        dataset = []
        with open(file_path, 'r') as data_file:
            for line in data_file:
                x, y, label = line.strip().split()
                dataset.append((float(x), float(y), int(label)))
        return dataset

    def split_dataset(self, dataset: list) -> tuple:
        random.shuffle(dataset)
        split_point = int(len(dataset) * 0.5)
        return dataset[:split_point], dataset[split_point:]

    def initialize_weights(self, dataset: list) -> np.ndarray:
        return np.full(len(dataset), 1.0 / len(dataset))

    def generate_classifiers(self, dataset: list, classifier_type: str) -> list:
        if classifier_type == 'line':
            # Possibly refine this to a more selective approach
            return list(combinations(dataset, 2))
        elif classifier_type == 'circle':
            # More selective than pairing every point with every other
            return list(combinations(dataset, 2))

    def predict(self, classifier, data, classifier_type):
        if classifier_type == 'line':
            point1, point2 = classifier
            cross_product = np.cross([point2[0] - point1[0], point2[1] - point1[1]], [data[0] - point1[0], data[1] - point1[1]])
            return 1 if cross_product > 0 else -1
        elif classifier_type == 'circle':
            center, radius_point = classifier
            # Make sure to exclude the label from both the center and radius_point when calculating distances
            center_coordinates = np.array(center[:2])  # Use only the x, y coordinates
            radius_point_coordinates = np.array(radius_point[:2])
            data_coordinates = np.array([data[0], data[1]])
            
            radius = np.linalg.norm(radius_point_coordinates - center_coordinates)
            distance = np.linalg.norm(data_coordinates - center_coordinates)
            
            return 1 if distance <= radius else -1


    def evaluate_classifier(self, classifier, weights, dataset, classifier_type):
        predictions = np.array([self.predict(classifier, data, classifier_type) for data in dataset])
        labels = np.array([data[2] for data in dataset])
        weighted_errors = weights * (predictions != labels)
        return np.sum(weighted_errors), predictions

    def update_weights(self, weights, predictions, labels, alpha):
        weights *= np.exp(-alpha * predictions * labels)
        weights /= np.sum(weights)  # Normalize
        return weights

    def aggregate_predictions(self, dataset, classifiers, alphas, classifier_type):
        final_predictions = np.zeros(len(dataset))
        for alpha, classifier in zip(alphas, classifiers):
            predictions = np.array([self.predict(classifier, data, classifier_type) for data in dataset])
            final_predictions += alpha * predictions
        return np.sign(final_predictions)

    def evaluate_performance(self, test_data, training_data, classifiers, alphas, classifier_type):
        training_labels = np.array([data[2] for data in training_data])
        test_labels = np.array([data[2] for data in test_data])
        
        training_predictions = self.aggregate_predictions(training_data, classifiers, alphas, classifier_type)
        test_predictions = self.aggregate_predictions(test_data, classifiers, alphas, classifier_type)
        
        empirical_error = np.mean(training_predictions != training_labels)
        true_error = np.mean(test_predictions != test_labels)
        
        return empirical_error, true_error


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


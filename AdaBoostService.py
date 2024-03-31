from matplotlib import pyplot as plt
import numpy as np
import random
from itertools import combinations

class AdaboostService:
    def __init__(self):
        pass

    def fetch_dataset(self, file_path: str) -> list:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                features = line.strip().split()
                data.append((float(features[0]), float(features[1]), int(features[2])))
        return data

    def split_data(self, dataset: list) -> tuple:
        random.shuffle(dataset)
        midpoint = int(len(dataset) * 0.5)
        return dataset[:midpoint], dataset[midpoint:]

    def set_initial_weights(self, data_points: list) -> dict:
        weight = 1.0 / len(data_points)
        return {point: weight for point in data_points}

    def craft_hypotheses(self, data_points: list):
        return list(combinations(data_points, 2))

    def evaluate_hypothesis(self, hypothesis, weights, data_points, shape):
        error = 0
        predictions = {}
        for point in data_points:
            guess = self.make_prediction(hypothesis, point, shape)
            predictions[point] = guess
            if guess != point[2]:
                error += weights[point]
        return error, predictions

    def make_prediction(self, hypothesis, point, shape):
        if shape == 'line':
            determinant = (hypothesis[1][0] - hypothesis[0][0]) * (point[1] - hypothesis[0][1]) - \
                          (hypothesis[1][1] - hypothesis[0][1]) * (point[0] - hypothesis[0][0])
            return 1 if determinant > 0 else -1
        elif shape == 'circle':
            radius = np.sqrt((hypothesis[1][0] - hypothesis[0][0])**2 + (hypothesis[1][1] - hypothesis[0][1])**2)
            distance = np.sqrt((point[0] - hypothesis[0][0])**2 + (point[1] - hypothesis[0][1])**2)
            return 1 if distance <= radius else -1

    def validate_model(self, test_data, top_hypotheses, alphas, shape):
        predictions = []
        for point in test_data:
            weighted_sum = sum(alpha * self.make_prediction(hypothesis, point, shape) 
                               for hypothesis, alpha in zip(top_hypotheses, alphas))
            prediction = 1 if weighted_sum > 0 else -1
            predictions.append((point, prediction))
        return predictions

    def calculate_model_accuracy(self, predictions):
        correct_predictions = sum(1 for point, prediction in predictions if prediction == point[2])
        return correct_predictions / len(predictions)
    
    def visualize(self, dataset, classifiers, classifier_type):
        # Set the figure size for the canvas
        plt.figure(figsize=(8, 6))  # You can adjust the width and height as needed

        # Convert dataset to a NumPy array for easier indexing
        dataset_np = np.array(dataset)

        # First, plot the dataset points with smaller dots
        for point in dataset_np:
            plt.scatter(point[0], point[1], color='red' if point[2] == 1 else 'blue', s=10)  # Adjust the size with `s`

        # Now plot the classifiers based on type
        if classifier_type == 'line':
            for classifier in classifiers:
                point1, point2 = classifier
                if point1[0] == point2[0]:  # Vertical line case
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

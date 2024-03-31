from matplotlib import pyplot as plt
import numpy as np
import random
from itertools import combinations

class AdaboostService:
    def __init__(self):
        pass

    def fetch_dataset(self, file_path: str) -> list:
        with open(file_path, 'r') as file:
            return [(float(x), float(y), int(label)) for x, y, label in (line.strip().split() for line in file)]

    def split_data(self, dataset: list) -> tuple:
        random.shuffle(dataset)
        midpoint = len(dataset) // 2
        return dataset[:midpoint], dataset[midpoint:]

    def set_initial_weights(self, data_points: list) -> dict:
        initial_weight = 1.0 / len(data_points)
        return {point: initial_weight for point in data_points}

    def craft_hypotheses(self, data_points: list):
        return list(combinations(data_points, 2))

    def evaluate_hypothesis(self, hypothesis, weights, data_points, shape):
        error, predictions = 0, {}
        for point in data_points:
            prediction = self.make_prediction(hypothesis, point, shape)
            predictions[point] = prediction
            if prediction != point[2]:
                error += weights[point]
        return error, predictions

    def make_prediction(self, hypothesis, point, shape):
        x_diff, y_diff = hypothesis[1][0] - hypothesis[0][0], hypothesis[1][1] - hypothesis[0][1]
        if shape == 'line':
            determinant = x_diff * (point[1] - hypothesis[0][1]) - y_diff * (point[0] - hypothesis[0][0])
            return 1 if determinant > 0 else -1
        elif shape == 'circle':
            radius = np.hypot(x_diff, y_diff)
            distance = np.hypot(point[0] - hypothesis[0][0], point[1] - hypothesis[0][1])
            return 1 if distance <= radius else -1

    def validate_model(self, test_data, top_hypotheses, alphas, shape):
        return [(point, 1 if sum(alpha * self.make_prediction(hypothesis, point, shape) for hypothesis, alpha in zip(top_hypotheses, alphas)) > 0 else -1) for point in test_data]

    def calculate_model_accuracy(self, predictions):
        correct_predictions = sum(1 for point, prediction in predictions if prediction == point[2])
        return correct_predictions / len(predictions)
    
    def visualize(self, dataset, classifiers, classifier_type):
        plt.figure(figsize=(8, 8))  # Square figure for better aspect ratio
        dataset_np = np.array(dataset)
        
        # Set the plot limits with some padding
        x_min, x_max = dataset_np[:, 0].min() - 1, dataset_np[:, 0].max() + 1
        y_min, y_max = dataset_np[:, 1].min() - 1, dataset_np[:, 1].max() + 1
        
        # Use a color map for the points
        colors = ['skyblue' if label == 1 else 'lightcoral' for _, _, label in dataset]
        plt.scatter(dataset_np[:, 0], dataset_np[:, 1], c=colors, s=50, edgecolors='w', linewidth=0.6)  # White edges around the points

        # Plot classifiers with decorative styles
        if classifier_type == 'line':
            for point1, point2 in classifiers:
                if point1[0] == point2[0]:  # Vertical line
                    plt.axvline(x=point1[0], color='mediumseagreen', linestyle=':', linewidth=2)
                else:
                    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                    intercept = point1[1] - slope * point1[0]
                    plt.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], color='mediumseagreen', linestyle=':', linewidth=2)
        elif classifier_type == 'circle':
            for center, point_on_circle in classifiers:
                radius = np.hypot(point_on_circle[0] - center[0], point_on_circle[1] - center[1])
                circle = plt.Circle(center, radius, color='violet', fill=False, linestyle=':', linewidth=2)
                plt.gca().add_artist(circle)

        # Enhance the plot with a title and axis labels using a fontdict
        font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 16}
        plt.title(f'AdaBoost Classifier: {classifier_type.capitalize()} Shape', fontdict=font)

        # Set the aspect of the plot to be equal
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')

        # Grid and legend can be added for better clarity
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.legend(['Classifier', 'Positive class', 'Negative class'], loc='best')

        # Finally, show the plot
        plt.show()
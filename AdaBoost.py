# Updated AdaBoost Experiment Implementation
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from AdaBoostService import AdaBoostService  # Adjusted import statement for the service class

class EnhancedAdaBoost:
    def __init__(self):
        self.service = AdaBoostService()  # Initialize with the AdaBoostService for utility functions

    def run_experiment(self, shape_type, total_runs=50, iterations=8, verbose=False, visualize=False, display_hypotheses=False):
        # Validate the shape type
        if shape_type not in ['line', 'circle']:
            print("Error: shape_type must be 'line' or 'circle'.")
            return

        empirical_errors_final, true_errors_final = np.zeros(iterations), np.zeros(iterations)

        # Perform AdaBoost runs
        for run in range(total_runs):
            dataset = self.service.load_data('circle_separator.txt')
            training_data, test_data = self.service.split_dataset(dataset)
            weights = self.service.initialize_weights(training_data)
            classifiers = self.service.generate_classifiers(training_data, shape_type)

            best_classifiers, alphas, empirical_errors, true_errors = [], [], [], []

            for i in range(iterations):
                best_classifier, lowest_error, best_predictions = None, float('inf'), None
                
                for classifier in classifiers:
                    error, predictions = self.service.evaluate_classifier(classifier, weights, training_data, shape_type)
                    
                    if error < lowest_error:
                        lowest_error, best_classifier, best_predictions = error, classifier, predictions
                
                alpha = 0.5 * np.log((1 - lowest_error) / (lowest_error + 1e-10))
                alphas.append(alpha)
                best_classifiers.append(best_classifier)
                
                # Update weights
                for data in training_data:
                    weights[data] *= np.exp(-alpha * best_predictions[data] * data[2])
                normalization_factor = sum(weights.values())
                weights = {data: weight / normalization_factor for data, weight in weights.items()}

                # Evaluate performance for the current iteration
                empirical_error, true_error = self.service.evaluate_performance(test_data, training_data, best_classifiers, alphas, shape_type)
                empirical_errors.append(empirical_error)
                true_errors.append(true_error)

                if verbose and run == total_runs - 1:  # Verbose output for the last run of each iteration
                    print(f'Iteration {i + 1}, Run {run + 1}: Empirical Error = {empirical_error:.8f}, True Error = {true_error:.8f}')

            # Aggregate the errors over all runs for each iteration
            empirical_errors_final += np.array(empirical_errors)
            true_errors_final += np.array(true_errors)

        # Average the errors over all runs
        empirical_errors_final /= total_runs
        true_errors_final /= total_runs

        if display_hypotheses:
            for idx, (hypothesis, alpha) in enumerate(zip(best_classifiers, alphas)):
                print(f"Hypothesis {idx + 1}: {hypothesis}, Alpha: {alpha}")

        if visualize:
            self.visualize(dataset, best_classifiers, shape_type)

        return empirical_errors_final, true_errors_final

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


        def evaluate_performance(self, test_data, training_data, classifiers, alphas, classifier_type):
            training_predictions = self.service.aggregate_predictions(training_data, classifiers, alphas, classifier_type)
            test_predictions = self.service.aggregate_predictions(test_data, classifiers, alphas, classifier_type)

            empirical_accuracy = self.service.calculate_accuracy(training_predictions)
            true_accuracy = self.service.calculate_accuracy(test_predictions)

            return 1 - empirical_accuracy, 1 - true_accuracy

# Main Execution
if __name__ == "__main__":
    # Example usage with specified configurations
    boost_classifier = EnhancedAdaBoost()
    current_time = datetime.now().strftime('%H:%M:%S')
    print(f"{current_time} Starting Enhanced AdaBoost experiment")

    # Run configuration
    shape_type = 'circle'  # line or 'circle'
    total_runs = 50
    iterations = 8
    verbose = False
    visualize = True
    display_hypotheses = False

    # Execute 
    empirical_errors, true_errors = boost_classifier.run_experiment(shape_type, total_runs, iterations, verbose, visualize, display_hypotheses)
    
    # Display final averaged results
    for i in range(iterations):
        print(f'Iteration {i+1}: Average Empirical Error = {empirical_errors[i]}, Average True Error = {true_errors[i]}')

# Updated AdaBoost Experiment Implementation
import numpy as np
from datetime import datetime
from AdaBoostService import AdaBoostService


class AdaBoost:
    def __init__(self):
        self.service = AdaBoostService()  # Initialize with the AdaBoostService for utility functions

    def run_experiment(self, data_path=str, shape_type='circle', total_runs=8, iterations=50, visualize=False):
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"{current_time} Starting AdaBoost experiment with configurations:")
        print(f"Shape Type: {shape_type}, Total Runs: {total_runs}, Iterations per Run: {iterations}, Visualization: {visualize}")

        if shape_type not in ['line', 'circle']:
            raise ValueError("Error: shape_type must be 'line' or 'circle'.")

        summed_empirical_errors, summed_true_errors = np.zeros(total_runs), np.zeros(total_runs)

        dataset = self.service.load_data(data_path)
        training_data, test_data = self.service.split_dataset(dataset)

        for run in range(total_runs):
            weights = self.service.initialize_weights(training_data)
            classifiers = self.service.generate_classifiers(training_data, shape_type)

            best_classifiers, alphas = [], []

            for iteration in range(iterations):
                best_classifier, lowest_error, best_predictions = None, float('inf'), None

                for classifier in classifiers:
                    error, predictions = self.service.evaluate_classifier(classifier, weights, training_data, shape_type)

                    if error < lowest_error:
                        lowest_error, best_classifier, best_predictions = error, classifier, predictions

                alpha = 0.5 * np.log((1 - lowest_error) / (lowest_error + 1e-10))
                alphas.append(alpha)
                best_classifiers.append(best_classifier)

                for i, data in enumerate(training_data):
                    prediction = best_predictions[i]
                    label = data[2]
                    weights[i] *= np.exp(-alpha * prediction * label)

                weights /= np.sum(weights)

            empirical_error, true_error = self.service.evaluate_performance(test_data, training_data, best_classifiers, alphas, shape_type)
            summed_empirical_errors[run], summed_true_errors[run] = empirical_error, true_error


            # Print the average errors of the last iteration across all runs
            for run in range(total_runs):
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"{current_time} Iteration {i+1}: Average Empirical Error = {summed_empirical_errors[run]}, Average True Error = {summed_true_errors[run]}")

            if visualize:
                self.service.visualize(dataset, best_classifiers, shape_type)



# Main Execution
if __name__ == "__main__":

    # Example usage with specified configurations

    shape_type = 'line'  # line or 'circle'
    total_runs = 1
    iterations = 8
    visualize = True
    
    # Execute 
    AdaBoost().run_experiment(data_path='circle_separator.txt', shape_type=shape_type, total_runs=total_runs, iterations=iterations, visualize=visualize)
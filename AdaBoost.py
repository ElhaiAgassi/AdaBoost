# Updated AdaBoost Experiment Implementation
import numpy as np
from datetime import datetime
from AdaBoostService import AdaBoostService  # Adjusted import statement for the service class

class AdaBoost:
    def __init__(self):
        self.service = AdaBoostService()  # Initialize with the AdaBoostService for utility functions

    def run_experiment(self, data_path=str, shape_type='circle', total_runs=8, iterations=50, visualize=False):
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"{current_time} Starting AdaBoost experiment with configurations:")
        print(f"Shape Type: {shape_type}, Total Runs: {total_runs}, Iterations per Run: {iterations}, Visualization: {visualize}")

        # Validate the shape type
        if shape_type not in ['line', 'circle']:
            raise ValueError("Error: shape_type must be 'line' or 'circle'.")

        # Arrays to store summed errors across all runs for averaging later
        summed_empirical_errors = np.zeros(total_runs)
        summed_true_errors = np.zeros(total_runs)

        dataset = self.service.load_data(data_path)
        training_data, test_data = self.service.split_dataset(dataset)

        for run in range(total_runs):
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

                # Update weights for the next iteration
                for data in training_data:
                    weight_update = np.exp(-alpha * best_predictions[data] * data[2])
                    weights[data] *= weight_update

                # Normalize weights
                normalization_factor = sum(weights.values())
                weights = {data: weight / normalization_factor for data, weight in weights.items()}

                empirical_error, true_error = self.service.evaluate_performance(test_data, training_data, best_classifiers, alphas, shape_type)
                empirical_errors.append(empirical_error)
                true_errors.append(true_error)

            # Sum errors of the last iteration of each run to aggregate
            summed_empirical_errors[run] = sum(empirical_errors)
            summed_true_errors[run] = sum(true_errors)

        # Calculate average errors over all runs
        avg_empirical_error = np.mean(summed_empirical_errors)
        avg_true_error = np.mean(summed_true_errors)

        # Print the average errors of the last iteration across all runs
        for i in range(total_runs):
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"{current_time} Iteration {i+1}: Average Empirical Error = {avg_empirical_error}, Average True Error = {avg_true_error}")

        if visualize:
            self.service.visualize(dataset, best_classifiers, shape_type)



# Main Execution
if __name__ == "__main__":

    # Example usage with specified configurations

    shape_type = 'circle'  # line or 'circle'
    total_runs = 8
    iterations = 50
    visualize = True
    
    # Execute 
    AdaBoost().run_experiment(data_path='circle_separator.txt', shape_type='circle', total_runs=8, iterations=50, visualize=True)

import numpy as np

from AdaBoostService import AdaboostService


class AdaBoost:
    def __init__(self):
        self.utility = AdaboostService()

    def execute_adaboost(self, shape_type, num_iterations,  visualize):
        data_processing = self.utility  # Streamlining method calls
        
        data_points = data_processing.fetch_dataset('circle_separator.txt')
        training_set, test_set = data_processing.split_data(data_points)
        weights = data_processing.set_initial_weights(training_set)
        possible_hypotheses = data_processing.craft_hypotheses(training_set)

        selected_hypotheses = []
        hypothesis_weights = []
        training_error_log = []
        test_error_log = []
        
        for _ in range(num_iterations):
            optimal_hypothesis = None
            minimal_error = float('inf')
            optimal_predictions = None
            
            for hypothesis in possible_hypotheses:
                error, predictions = data_processing.evaluate_hypothesis(hypothesis, weights, training_set, shape_type)
                if error < minimal_error:
                    minimal_error = error
                    optimal_hypothesis = hypothesis
                    optimal_predictions = predictions

            alpha = 0.5 * np.log((1 - minimal_error) / (minimal_error + 1e-10))
            hypothesis_weights.append(alpha)
            selected_hypotheses.append(optimal_hypothesis)
            
            # Update weights for misclassified points
            for point in training_set:
                weights[point] *= np.exp(-alpha * optimal_predictions[point] * point[2])
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {point: weight / total_weight for point, weight in weights.items()}
            
            # Log errors for analysis
            test_error = 1 - data_processing.calculate_model_accuracy(data_processing.validate_model(test_set, selected_hypotheses, hypothesis_weights, shape_type))
            test_error_log.append(test_error)

            training_error = 1 - data_processing.calculate_model_accuracy(data_processing.validate_model(training_set, selected_hypotheses, hypothesis_weights, shape_type))
            training_error_log.append(training_error)
        
        if visualize:
                self.utility.visualize(data_points, selected_hypotheses, shape_type)
        
        final_training_error = 1 - data_processing.calculate_model_accuracy(data_processing.validate_model(training_set, selected_hypotheses, hypothesis_weights, shape_type))
        final_test_error =  1 - data_processing.calculate_model_accuracy(data_processing.validate_model(test_set, selected_hypotheses, hypothesis_weights, shape_type))
        
        # Return the accumulated errors and the error logs for each iteration
        return final_training_error, final_test_error, training_error_log, test_error_log

# Implementation example
if __name__ == "__main__":
    AdaBoost = AdaBoost()
    
    # Configuration for the experiment
    total_runs = 50
    iterations_per_run = 8
    classifier_shape = 'line'  # Choose 'line' or 'circel' classifiers
    dataset_path = 'circle_separator.txt'
    visualize = True

    print(f"Running AdaBoost with total_runs={total_runs}, iterations_per_run={iterations_per_run}, "
          f"classifier_shape='{classifier_shape}', dataset_path='{dataset_path}', "
          f"visualize={'Yes' if visualize else 'No'}")
    
    # Aggregate error metrics
    aggregated_training_error = aggregated_test_error = 0 
    average_training_errors_per_iteration = np.zeros(iterations_per_run)
    average_test_errors_per_iteration = np.zeros(iterations_per_run)

    # Run the adaboost multiple times
    for _ in range(total_runs):
        training_error, test_error, training_errors, test_errors = AdaBoost.execute_adaboost(
            classifier_shape, iterations_per_run, visualize)
        
        aggregated_training_error += training_error
        aggregated_test_error += test_error
        average_training_errors_per_iteration += np.array(training_errors)
        average_test_errors_per_iteration += np.array(test_errors)

    # Calculate and print average errors
    average_training_errors_per_iteration /= total_runs
    average_test_errors_per_iteration /= total_runs
    avg_training_error = aggregated_training_error / total_runs
    avg_test_error = aggregated_test_error / total_runs

    for i in range(iterations_per_run):
        print(f'Iteration {i+1}: Avg True Training Error = {average_training_errors_per_iteration[i]}, Avg Empirical Test Error = {average_test_errors_per_iteration[i]}')
    
    print(f"Average Training Error (across all runs): {avg_training_error}")
    print(f"Average Test Error (across all runs): {avg_test_error}")

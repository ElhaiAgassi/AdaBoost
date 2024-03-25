import matplotlib.pyplot as plt
import numpy as np
import logging

from AdaBoostService import AdaBoostService

# Setup basic logging for monitoring the progress
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def multiple_adaboost_runs(file_path, runs=50, T=8, classifier_type='line'):
    print(f"Running multiple adaboost runs: {T} AdaBoost rounds, {runs} iterations each, type: {classifier_type}")

    all_empirical_errors = np.zeros(T)  # Accumulate errors for averaging
    all_true_errors = np.zeros(T)

    for run in range(runs):
        # Load and split data anew for each run
        train_data, test_data = AdaBoostService.load_and_split_data(file_path)
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        # Reset models and alphas for each run
        models, alphas = [], []
        weights = np.full(len(X_train), 1/len(X_train))  # Initialize weights

        for k in range(T):
            hypotheses = AdaBoostService.generate_hypotheses(X_train, classifier_type)
            best_hypothesis, best_error, best_predictions = None, np.inf, None
            for hypothesis in hypotheses:
                error, predictions = AdaBoostService.evaluate_hypothesis(hypothesis, X_train, y_train, weights, classifier_type)
                if error < best_error:
                    best_error = error
                    best_hypothesis = hypothesis
                    best_predictions = predictions

            alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
            models.append(best_hypothesis)
            alphas.append(alpha)
            weights = AdaBoostService.update_weights(weights, alpha, y_train, best_predictions)

            # Calculate errors for this round
            empirical_error = AdaBoostService.calculate_error(y_train, AdaBoostService.predict(models, alphas, X_train, classifier_type))
            true_error = AdaBoostService.calculate_error(y_test, AdaBoostService.predict(models, alphas, X_test, classifier_type))

            # Accumulate errors across runs for averaging
            all_empirical_errors[k] += empirical_error
            all_true_errors[k] += true_error

    # Calculate average errors over all runs
    average_empirical_errors = all_empirical_errors / runs
    average_true_errors = all_true_errors / runs

    # Printing performance summary
    print("AdaBoost Performance Summary:")
    for round_num in range(T):
        print(f"Round {round_num + 1}: Average Empirical Error: {average_empirical_errors[round_num]:.4f}, Average True Error: {average_true_errors[round_num]:.4f}")

    return X_train, y_train, models, classifier_type

# Example usage
if __name__ == "__main__":
    file_path = "circle_separator.txt"
    classifier_type = 'circle'
    X_train, y_train, models, classifier_type = multiple_adaboost_runs(file_path, 50, 8, classifier_type)
    AdaBoostService.visualize_best_hypotheses(X_train, y_train, models, classifier_type)

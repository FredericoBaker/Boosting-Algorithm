import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time

import utils
import adaboost


def cross_validation_adaboost(X, y, fold_indices, n_iterations):
    """
    Perform k-fold cross-validation for AdaBoost and return the mean and standard deviation of errors.
    """
    errors = []

    for train_index, val_index in fold_indices:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = adaboost.AdaBoost(n_iterations=n_iterations)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = np.sum(np.equal(y_val, y_pred)) / len(y_val)
        error = 1 - accuracy
        errors.append(error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    return mean_error, std_error

def eval_adaboost(X_train, X_test, y_train, y_test, n_iterations):
    """
    Evaluate AdaBoost on the test set and return the error.
    """
    model = adaboost.AdaBoost(n_iterations)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_test)
    error = 1 - accuracy

    return error

def load_and_clean_data(filepath):
    """
    Load and clean the dataset.
    """
    columns = [i for i in range(1, 10)]
    columns.append("target")
    data = pd.read_csv(filepath, names=columns)

    # Define the mapping for replacements
    replace_map = {'x': 1, 'o': -1, 'b': 0, 'positive': 1, 'negative': -1}

    # Apply the mapping to each column
    for col in data.columns:
        data[col] = data[col].map(replace_map).astype(int)

    return data

def main(filepath, test_size, k_folds, max_iterations):
    # Load and preprocess data
    data = load_and_clean_data(filepath)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=test_size, random_seed=42)

    # Generate fold indices for cross-validation
    fold_indices = utils.kfold_indices(X_train, k_folds, random_seed=42)

    # Perform cross-validation and evaluation
    errors_means_train = []
    errors_stds_train = []
    errors_test = []
    time_to_train = []
    for n_iterations in tqdm(range(1, max_iterations + 1), desc ="Progress"):
        start = time.time()
        mean_error, std_error = cross_validation_adaboost(X_train, y_train, fold_indices, n_iterations)
        time_to_train.append(time.time() - start)

        errors_means_train.append(mean_error)
        errors_stds_train.append(std_error)

        test_error = eval_adaboost(X_train, X_test, y_train, y_test, n_iterations)
        errors_test.append(test_error)

    # Plot the errors
    utils.plot_errors(errors_means_train, errors_stds_train, errors_test)
    utils.plot_time(time_to_train)

    print(f"Last error: {errors_test[-1]}")

    min_val_error = min(errors_means_train)
    acceptable_error_threshold = min_val_error * 1.05

    for i, score in enumerate(errors_means_train):
        if score <= acceptable_error_threshold:
            best_n_estimators = i
            break
    print(f"Best number of estimators: {best_n_estimators}")
    best_test_error = eval_adaboost(X_train, X_test, y_train, y_test, best_n_estimators)
    print(f"Best estimators error: {best_test_error}")
    print(f"Best estimators accuracy: {1 - best_test_error}")

    print("Evaluation process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaBoost Cross-Validation")
    parser.add_argument('--filepath', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of boosting iterations')

    args = parser.parse_args()

    main(filepath=args.filepath, test_size=args.test_size, k_folds=args.k_folds, max_iterations=args.max_iterations)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kfold_indices(data, k, random_seed=42):
    """
    Generate k-fold indices for cross-validation.

    Returns:
    - folds: list of tuples, each containing train and test indices
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop
    
    return folds

def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Split the data into training and testing sets.
    """
    np.random.seed(random_seed)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    np.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def plot_errors(train_means, train_stds, test_errors, save_path='errors_plot.png'):
    """
    Plot training and test errors over iterations and save the plot to a specified path.
    """
    iterations = range(1, len(train_means) + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, train_means, label='Mean Training Error')
    plt.fill_between(iterations, np.array(train_means) - np.array(train_stds), np.array(train_means) + np.array(train_stds), alpha=0.2)
    
    plt.plot(iterations, test_errors, label='Test Error')
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training and Test Errors over Iterations')
    
    # Save the plot
    plt.savefig(save_path)

def plot_time(time_to_train, save_path='elapsed_time_plot.png'):
    """
    Plot elapsed time to train and save the plot to a specified path.
    """
    iterations = range(1, len(time_to_train) + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, time_to_train, label='Time to train')    
    plt.xlabel('Number of Iterations')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title('Elapsed time to train')
    
    # Save the plot
    plt.savefig(save_path)

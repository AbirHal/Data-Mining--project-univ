import numpy as np
from metrics import gini, entropy,r2_score
import matplotlib.pyplot as plt
import pandas as pd



class DecisionTree:
    


    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        X = self._ensure_numpy_array(X)
        y = self._ensure_numpy_array(y)

        self.tree = self._build_tree(X, y)

    def predict(self, X):
        X = self._ensure_numpy_array(X)
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _ensure_numpy_array(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        else:
           data=np.array(data)           
        return data




        
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check termination criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < 2:
            return {'leaf': True, 'class': np.argmax(np.bincount(y))}

        # Determine split
        if self.criterion == 'gini':
            gain = self._gini_gain
        elif self.criterion == 'entropy':
            gain = self._entropy_gain
        else:
            raise ValueError('Unknown criterion:', self.criterion)

        best_gain = -1
        for feature_idx in range(n_features):
            for threshold in np.unique(X[:, feature_idx]):
                # Split data
                left_idx = X[:, feature_idx] < threshold
                left_X, left_y = X[left_idx], y[left_idx]
                right_X, right_y = X[~left_idx], y[~left_idx]

                # Calculate information gain
                gain_val = gain(y, left_y, right_y)

                # Check if this is the best split so far
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_split = {'feature_idx': feature_idx, 'threshold': threshold, 'left_X': left_X,
                                  'left_y': left_y, 'right_X': right_X, 'right_y': right_y}
    
        # Create sub-trees
        if best_gain > 0:
            left_tree = self._build_tree(best_split['left_X'], best_split['left_y'], depth + 1)
            right_tree = self._build_tree(best_split['right_X'], best_split['right_y'], depth + 1)
            return {'leaf': False, 'feature_idx': best_split['feature_idx'], 'threshold': best_split['threshold'],
                    'left': left_tree, 'right': right_tree}
        else:
            return {'leaf': True, 'class': np.argmax(np.bincount(y))}

    def _predict(self, inputs, tree):
        if tree['leaf']:
            return tree['class']
        else:
            if inputs[tree['feature_idx']] < tree['threshold']:
                return self._predict(inputs, tree['left'])
            else:
                return self._predict(inputs, tree['right'])

    def _gini(self, y):
        return gini(y)

    def _entropy(self, y):
        return entropy(y)

    def _gini_gain(self, y, left_y, right_y):
      p = len(left_y) / len(y)
      return self._gini(y) - p * self._gini(left_y) - (1 - p) * self._gini(right_y)
    def _entropy_gain(self, y, left_y, right_y):
      n = len(y)
      n_left = len(left_y)
      n_right = len(right_y)

    # Calculate parent entropy
      parent_entropy = self._entropy(y)

    # Calculate weighted average of child entropies
      child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)

    # Calculate entropy gain
      entropy_gain = parent_entropy - child_entropy
      return entropy_gain

    def plot(self):
        plt.figure(figsize=(10, 6))
        self._plot_tree(self.tree, depth=0)
        plt.axis('off')
        plt.show()

    def _plot_tree(self, node, depth, x=0, y=0, parent_x=0, parent_y=0, branch=''):
        # Set node style
        box_style = dict(boxstyle='round', facecolor='lightgray')
        arrow_style = dict(arrowstyle='->', color='black')

        if node['leaf']:
            class_info = node['class']
            if isinstance(class_info, dict):
                class_counts = class_info
                class_labels = list(class_counts.keys())
                class_probabilities = [class_counts[label] / sum(class_counts.values()) for label in class_labels]
                class_text = '\n'.join([f'Class {label}: {count} ({prob:.2%})' for label, count, prob in zip(class_labels, class_counts.values(), class_probabilities)])
            elif isinstance(class_info, list) or isinstance(class_info, np.ndarray):
                class_counts = dict(enumerate(class_info))
                class_labels = list(class_counts.keys())
                class_probabilities = [class_counts[label] / sum(class_counts.values()) for label in class_labels]
                class_text = '\n'.join([f'Class {label}: {count} ({prob:.2%})' for label, count, prob in zip(class_labels, class_counts.values(), class_probabilities)])
            else:
                class_text = f'Class: {class_info}'

            plt.annotate(class_text, xy=(x, y), xytext=(x, y - 0.1),
                         bbox=dict(boxstyle='round', facecolor='white'), ha='center')
        else:
            plt.annotate(f'Feature {node["feature_idx"]}\n<{node["threshold"]}', xy=(x, y), xytext=(x, y - 0.1),
                         bbox=box_style, arrowprops=arrow_style, ha='center')
            left_x = x - 0.5 / 2 ** (depth + 1)
            right_x = x + 0.5 / 2 ** (depth + 1)
            left_y = y - 0.2
            right_y = y - 0.2

            plt.plot([x, left_x], [y - 0.05, left_y + 0.05], color='black')
            plt.plot([x, right_x], [y - 0.05, right_y + 0.05], color='black')

            self._plot_tree(node['left'], depth + 1, left_x, left_y)
            self._plot_tree(node['right'], depth + 1, right_x, right_y)




class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = self._ensure_numpy_array(X)
        y = self._ensure_numpy_array(y)

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            y_pred = self._predict(X)
            error = y_pred - y

            # Update weights and bias
            gradient_weights = np.dot(X.T, error) / len(X)
            gradient_bias = np.mean(error)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        X = self._ensure_numpy_array(X)
        return self._predict(X)

    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def _ensure_numpy_array(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data

    def _r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)    


class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for sample in X:
            distances = self._euclidean_distance(sample, self.X_train)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            y_pred.append(unique_labels[np.argmax(counts)])
        return np.array(y_pred)

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))


def cross_validation(X, y, n_folds, knn_params):
    fold_size = len(X) // n_folds
    scores = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        X_test = X[start:end]
        y_test = y[start:end]

        knn = KNN(**knn_params)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy(y_test, y_pred)
        scores.append(score)

    return np.mean(scores)


def grid_search(X, y, n_folds, knn_params_list):
    best_params = None
    best_score = -1
    for knn_params in knn_params_list:
        score = cross_validation(X, y, n_folds, knn_params)
        if score > best_score:
            best_score = score
            best_params = knn_params

    return best_params, best_score


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)



class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, criterion='gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.estimators = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for _ in range(self.n_estimators):
            # Randomly select a subset of samples (with replacement)
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[sample_indices]
            y_subset = y[sample_indices]

            # Create a decision tree and fit it to the subset of data
            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion)
            tree.fit(X_subset, y_subset)

            # Add the decision tree to the list of estimators
            self.estimators.append(tree)

    def predict(self, X):
        X = np.array(X)
        # Make predictions using each decision tree and return the majority vote
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        y_pred = np.median(predictions,axis=0).astype(int)
        return np.mean(predictions, axis=0).astype(int)   

   
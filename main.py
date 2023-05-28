import numpy as np

#from hodhoud import DecisionTree

from hodhoud import  DecisionTree,LinearRegression,KNN,grid_search



# Example usage 2:
X_train2 = [[1,0,1,0], [0,1,0,1], [1,1,1,0], [1,0,1,1], [0,1,1,1],[0,1,0,0], [1,0,0,1], [1,1,0,0]]
y_train2 = [1,1,1,1,0,0,0,0]
#X_train2 = [[1,1], [0,1], [1,0], [0,0]]
#y_train2 = [0,1,0,1]
#X_train = np.asarray(X_train)
#y_train = np.asarray(y_train)

clf = DecisionTree(criterion='gini', max_depth=4)
print("***************DecisionTree  Module**************")

entropy = clf._entropy(y_train2)
print("Entropy:", entropy)
gini = clf._gini(y_train2)
print("Gini Index:", gini)
clf.fit(X_train2, y_train2)


#X_test = [[0, 1], [0, 1], [1, 0], [1, 1]]
#X_test = np.asarray(X_test)
#X_test = [[1,0]]
X_test = [[1,1,0,0]]


y_pred = clf.predict(X_test)

print("predictions of DecisionTree",y_pred)


clf.plot()


# Example usage:
print("***************LinearRegression  Module**************")

X_train = np.array([[1], [2], [3], [5], [6]])
y_train = np.array([2, 3.5, 4, 6.7,7.3])

# Create an instance of LinearRegression
regressor = LinearRegression()

# Fit the model
regressor.fit(X_train, y_train)

# Make predictions
X_test = np.array([[7]])
y_pred = regressor.predict(X_test)

y_true = np.array([12, 14, 16])
r2 = regressor._r2_score(y_true, y_pred)

print("predictions of LinearRegression",y_pred)
print("R2 score:", r2)

print("*************** KNN  Module**************")


# Example usage:
X_train = [[1, 1], [2, 2], [3, 3], [4, 4]]
y_train = [0, 0, 1, 1]
# Convert X and y to numpy arrays
X = np.array(X_train)
y = np.array(y_train)
knn_params_list = [
    {'n_neighbors': 3},
    {'n_neighbors': 5},
    {'n_neighbors': 7}
]

knn = KNN(n_neighbors=3)
knn.fit(X_train, y_train)

X_test = [[2.5, 2.5], [3.5, 3.5]]
y_pred = knn.predict(X_test)


best_params, best_score = grid_search(X, y, n_folds=3, knn_params_list=knn_params_list)

print("Best parameters:", best_params)
print("Best score:", best_score)

print("predictions of KNN",y_pred)

'''''
print("*************** RandomForest Module**************")

# Example usage:

X_train = [[1, 1], [0, 1], [1, 0], [0, 0]]
y_train = [0, 1, 0, 1]

rf = RandomForest(n_estimators=100, max_depth=2, criterion='gini')
rf.fit(X_train, y_train)

X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_pred = rf.predict(X_test)

print("predictions of RandomForest",y_pred)

'''






import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

"""Preprocessing"""

# Load dataset
file_path = "baseball.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Define original numeric features
numeric_features = ['YRS', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS', 'BA']

# Perform feature analysis
features_to_analyze = numeric_features

# Compute correlation matrix
corr_matrix = df[features_to_analyze].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()
print("Correlation matrix saved as correlation_matrix.png")

# Compute correlation with HOF
correlation_with_target = df[features_to_analyze].corrwith(df['HOF']).abs()
correlation_df = correlation_with_target.sort_values(ascending=False).reset_index()
correlation_df.columns = ["Feature", "Correlation"]

# Save correlation with HOF plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Correlation", y="Feature", data=correlation_df, legend=False)
plt.title("Feature Correlation with HOF")
plt.savefig("feature_correlation.png")
plt.close()
print("Feature correlation with HOF plot saved as feature_correlation.png")

# Identify highly correlated features (above 0.9)
high_corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
for col in high_corr_pairs.columns:
    high_corr = high_corr_pairs[col][high_corr_pairs[col] > 0.9].index.tolist()
    if high_corr:
        print(f"Feature '{col}' is highly correlated with: {high_corr}")

# Identify features with low correlation to HOF (below 0.1)
low_corr_features = correlation_df[correlation_df['Correlation'] < 0.1]['Feature'].tolist()
print("Features with low correlation to HOF:", low_corr_features)

# Drop irrelevant features
print("dropping irrelevant features:")
features_to_drop = ['PLAYER']
print(features_to_drop)
df.drop(features_to_drop, axis=1, inplace=True)

# Normalize data
scaler = MinMaxScaler()
print(f"Normalizing with {scaler}...")
numeric_cols = [col for col in df.columns if col not in ['HOF']]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save dataset
filename = "baseball_normalized.csv"
df.to_csv(filename, index=False)
print(f"Dataset saved as {filename}")


"""Load dataset"""

# Load dataset: X is the points, y is the labels
file_path = "baseball_normalized.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")
X = df.drop("HOF", axis=1, inplace=False)
y = df['HOF']

# Split into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


"""K-NN"""

# Find best hyperparameters
param_grid = {'n_neighbors': range(1, 21, 2)}
model = KNeighborsClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
print(f"Best k found: {best_k}")

# Create and train model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


"""Logistic Regression"""

# Find best hyperparameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 6, 10, 20, 50],  # Regularization strength
    'solver': ['lbfgs', 'saga', 'liblinear', 'newton-cg'],  # Different solvers
    'penalty': ['l2'],  # Regularization type (avoid 'l1' unless using liblinear)
    'max_iter': [500, 1000, 2000],  # Iterations for convergence
    'tol': [1e-4, 1e-3, 1e-2],  # Tolerance
    'class_weight': ['balanced', None]  # Handles class imbalance
}
model = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Create and train model
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


"""Random Forest"""

# Best hyperparameters were found  with grid search and trial-and-error.
# The grid search for the Random Forest hyperparameters takes much more time than for the other algorithms

# Create and train model
model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=25,
                               max_features='sqrt', min_samples_split=10,
                               min_samples_leaf=1, class_weight=None)
model.fit(X_train, y_train)
# Test model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns,
                                   'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance',
                                                    ascending=False)

# Save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance in Random Forest")
plt.savefig("rf_feature_importance.png")
plt.close()
print("Feature importance plot saved as rf_feature_importance.png")
print(feature_importance)


"""SVM"""

# Find best hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly'],
    'degree': [2, 3, 4],  # Only applies to polynomial kernel
    'class_weight': [{0: 1, 1: w} for w in [2, 2.5, 3]]
}
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

# Create and train model
model = grid.best_estimator_
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
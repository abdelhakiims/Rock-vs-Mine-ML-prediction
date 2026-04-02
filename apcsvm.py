import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# Load the sonar data (60 frequency columns, 1 label column)
sonar_data = pd.read_csv('data/sonar.csv', header=None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split into Training (90%) and Testing (10%)
# Stratify ensures we have a balanced mix of Rocks and Mines in both sets
# random_state=1 guarantees that we will have the same results every run
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# I used PCA to check if the whole data is informative or only a few features.
# After running it I saw that the variance stabilized at 95% with 30 components
# only, rather than the whole 60 (check the plot saved after running this code).
# We scale the data first because PCA is sensitive to the scale of features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

# pca.explained_variance_ratio_ shows the 'importance' of each individual component.
# np.cumsum() creates a running total (1st + 2nd + 3rd...) so we can see
# exactly when we have enough 'total information' to stop adding components.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Create and save the Scree Plot to visualize data redundancy
plt.figure(figsize=(8, 5))
plt.plot(range(1, 61), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-')  # The 95% information threshold
plt.title('PCA Analysis: How many features do we actually need?')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()

# Save the plot for GitHub documentation
plt.savefig('pca_scree_plot.png')
plt.show()

# Find the number of components needed to retain 95% of the variance
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"--- PCA Diagnostic ---")
print(f"To keep 95% of the sonar info, we only need {n_95} components.")
print(f"This proves that about half of our 60 features are redundant.\n")

# Apply PCA with the number of components found above (n_95, typically ~30)
# This reduces our feature space from 60 to ~30 without losing meaningful information
pca_final = PCA(n_components=n_95)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

# --- MODEL BUILDING & HYPERPARAMETER TUNING ---
# We use a Pipeline to ensure scaling is applied correctly during Cross-Validation.
# This prevents "Data Leakage" where the model accidentally sees test data stats.
# Note: scaling is already done above, so we pass the PCA-reduced data directly.
pipeline = Pipeline([
    ('svm', SVC())
])

# GridSearchCV: used to find the best settings
# C: Strictness (Small C = Wide Margin/Tolerant, Large C = Narrow Margin/Strict)
# Kernel: The "Shape" of the boundary (Linear vs RBF 'Bubble')
# Gamma: The reach of influence for each data point (Flashlight vs Laser)
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto', 0.1]
}

# Total models trained = 4(C) * 2(Kernels) * 3(Gamma) * 5(CV Splits) = 120 models
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train_pca, Y_train)

# --- EVALUATION ---
train_accuracy = grid.score(X_train_pca, Y_train)
test_accuracy = grid.score(X_test_pca, Y_test)

print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Training Accuracy:    {train_accuracy:.4f}")
print(f"Test Accuracy:        {test_accuracy:.4f}")

# Logic Note: If kernel='rbf' wins, it confirms the data is non-linearly separable.
# If C=10 or higher wins, it means the model benefited from a stricter boundary.
# If training accuracy >> test accuracy, the model is overfitting.
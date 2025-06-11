"""
Fault Prediction Model Training Script

Purpose:
This script trains a Random Forest classifier to predict network faults based on Key Performance Indicators (KPIs).
The process includes:
1.  Data Loading: Loads KPI data from a CSV file.
2.  Exploratory Data Analysis (EDA): Includes placeholders for initial data inspection.
3.  Feature Engineering: Creates a binary target variable 'fault_flag'.
4.  Data Preprocessing: Selects features, splits data into training/test sets, and applies StandardScaler.
5.  Hyperparameter Tuning: Uses GridSearchCV to find the best hyperparameters for the RandomForestClassifier.
6.  Feature Importance Analysis: Extracts, prints, and plots feature importances from the best model.
7.  Model Evaluation: Evaluates the best model using a classification report and ROC AUC score.
8.  Model Saving: Saves the trained model and the feature importance plot.

Prerequisites:
- Ensure `data/kpi_sample.csv` exists in the specified path and is correctly formatted.
- Ensure all required libraries (e.g., pandas, scikit-learn, matplotlib) are installed.
  It's recommended to use a virtual environment and install packages from a `requirements.txt` file if available.
"""

# === Import Libraries ===
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os

# === Configuration (Optional) ===
# You can define global configurations here, e.g., random_state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_FILEPATH = "data/kpi_sample.csv"
MODEL_OUTPUT_PATH = "models/fault_predictor.pkl"
REPORTS_DIR = "reports"
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(REPORTS_DIR, 'feature_importances.png')

# === Step 1: Data Loading ===
print(f"Loading data from {DATA_FILEPATH}...")
df = pd.read_csv(DATA_FILEPATH)
print("Data loaded successfully.")

# === Step 2: Exploratory Data Analysis (Placeholders) ===
# --- Initial Data Inspection ---
# It's crucial to understand your data. Uncomment and run these lines.
# print("\n--- Data Head ---")
# print(df.head())
# print("\n--- Data Info ---")
# print(df.info())
# print("\n--- Data Description ---")
# print(df.describe())
# print("\n--- Missing Values ---")
# print(df.isnull().sum())

# --- Target Variable Exploration ---
# This step is done after fault_flag creation, but EDA for it starts here.
# print("\n--- Fault Flag Value Counts (Class Balance) ---")
# This will be printed after 'fault_flag' is created.

# --- Guidance for Detailed EDA ---
# Consider more detailed EDA:
# - Visualizations (histograms, box plots for numerical features, count plots for categorical).
# - Correlation analysis between features and with the target variable.
# - Outlier detection and handling strategies.
# Based on EDA, you might need further data cleaning or feature engineering.

# === Step 3: Feature Engineering ===
# --- Create Target Column ---
# Create a binary target column 'fault_flag' (1 = fault, 0 = no fault)
# This is based on the 'call_drops' KPI; adjust logic if your fault definition differs.
print("\nCreating 'fault_flag' target variable...")
df['fault_flag'] = df['call_drops'].apply(lambda x: 1 if x > 0 else 0)
print("'fault_flag' created.")

# --- Post-Feature Engineering EDA ---
print("\n--- Fault Flag Value Counts (Class Balance) ---")
print(df['fault_flag'].value_counts(normalize=True))
# Note: If 'fault_flag' is highly imbalanced (check with value_counts()),
# consider using techniques like SMOTE (Synthetic Minority Over-sampling Technique)
# to improve model performance on the minority class.
# This would require installing the 'imbalanced-learn' library:
# `pip install imbalanced-learn`
#
# Example SMOTE usage (apply only to training data after splitting and scaling):
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=RANDOM_STATE)
# X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# Then, train your model on this resampled data.

# === Step 4: Data Preprocessing ===
# --- Feature Selection ---
# Define features (X) and target (y)
print("\nSelecting features and target...")
features = ['RSRP', 'RSRQ', 'SINR', 'throughput_Mbps'] # Adjust if your features differ
X = df[features]
y = df['fault_flag']
print(f"Selected features: {features}")

# --- Train-Test Split ---
print(f"\nSplitting data into training and test sets (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Data split successfully.")

# --- Feature Scaling ---
# Scaling is important for distance-based algorithms and helps gradient descent converge faster.
# RandomForest is not strictly sensitive to feature scaling, but it doesn't hurt.
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
# Fit the scaler on the training data only to prevent data leakage from the test set.
X_train_scaled = scaler.fit_transform(X_train)
# Apply the same fitted scaler to transform the test data.
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

# === Step 5: Model Training and Hyperparameter Tuning ===
print("\nStarting model training and hyperparameter tuning with GridSearchCV...")
# --- Define Parameter Grid for GridSearchCV ---
# These are common hyperparameters for RandomForest.
# User can expand these lists/ranges for a more exhaustive search, but be mindful of training time.
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20],   # Maximum depth of the tree
    'min_samples_split': [2, 5],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],    # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
    'class_weight': [None, 'balanced'] # To handle class imbalance. 'balanced' automatically adjusts weights.
}
print(f"Parameter grid for GridSearchCV: {param_grid}")

# --- Initialize RandomForestClassifier ---
# The estimator for GridSearchCV. random_state is set for reproducibility.
# Other hyperparameters will be tuned by GridSearchCV.
rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

# --- Initialize and Fit GridSearchCV ---
# cv=5 means 5-fold cross-validation.
# scoring='roc_auc' is suitable for binary classification and often preferred for imbalanced datasets.
# n_jobs=-1 uses all available CPU cores for parallel processing.
# verbose=1 provides progress messages.
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Fitting GridSearchCV... This may take some time.")
grid_search.fit(X_train_scaled, y_train)
print("GridSearchCV fitting complete.")

# --- Retrieve the Best Estimator ---
best_model = grid_search.best_estimator_
print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best ROC AUC score during cross-validation: {grid_search.best_score_:.4f}")

# === Step 6: Feature Importance Analysis ===
print("\nPerforming feature importance analysis...")
importances = best_model.feature_importances_
feature_importance_series = pd.Series(importances, index=features).sort_values(ascending=False)

print("\nFeature Importances (from best model):")
print(feature_importance_series)

# --- Plotting Feature Importances ---
print("\nPlotting feature importances...")
plt.figure(figsize=(10, 6))
feature_importance_series.plot(kind='barh', color='skyblue')
plt.title('Feature Importances for Fault Prediction (Best Model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.tight_layout()

# --- Save the Plot ---
# Ensure the reports directory exists
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"Created directory: {REPORTS_DIR}")

plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH)
print(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PLOT_PATH}")
plt.close()  # Close the plot to free up memory

# === Step 7: Model Evaluation ===
print("\nEvaluating the best model on the test set...")
# --- Make Predictions ---
# Ensure to use scaled features for prediction
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class (for ROC AUC)

# --- Calculate and Print Metrics ---
print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score (Best Model): {roc_auc:.4f}")

# === Step 8: Save the Best Trained Model ===
print(f"\nSaving the best trained model to: {MODEL_OUTPUT_PATH}...")
# Ensure the models directory exists (if MODEL_OUTPUT_PATH includes a directory)
model_dir = os.path.dirname(MODEL_OUTPUT_PATH)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

joblib.dump(best_model, MODEL_OUTPUT_PATH)
print(f"✅ Model saved successfully to {MODEL_OUTPUT_PATH}")

print("\n--- Script Execution Finished ---")

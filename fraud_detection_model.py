import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('E:/sentinel-gateway/large_transaction_data.csv')
print(f"Columns in dataset: {data.columns}")
print(f"Total records: {len(data)}")

# Encode categorical features
data = pd.get_dummies(data, columns=['location', 'transaction_type'], drop_first=True)

# Features and target
X = data.drop('fraud', axis=1)
y = data['fraud']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Check class balance before SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts())

# Check class balance after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_model, 'E:/sentinel-gateway/fraud_detection_model.joblib')
print("Model saved successfully")

# Cross-validation
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV accuracy: {cv_scores.mean() * 100:.2f}%')

# Predictions
y_pred = best_model.predict(X_test)

# Confusion matrix and report
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', classification_report(y_test, y_pred))

from joblib import dump
dump(best_model, 'E:/sentinel-gateway/model 111/fraud_detection_model.joblib')
print("Model saved successfully!")

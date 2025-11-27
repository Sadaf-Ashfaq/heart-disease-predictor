import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heart.csv')

print("\n=== Initial Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ===== DATA CLEANING =====
print("\n=== DATA CLEANING ===")

# Handle missing values
print(f"Missing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# Create binary target variable (0: no disease, 1: disease)
df['target'] = (df['num'] > 0).astype(int)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print(f"\nData shape after cleaning: {df.shape}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# ===== EXPLORATORY DATA ANALYSIS =====
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation analysis
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Correlation matrix saved as 'correlation_matrix.png'")

# Target distribution
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Heart Disease (0: No Disease, 1: Disease)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("Target distribution saved as 'target_distribution.png'")

# Age distribution by target
plt.figure(figsize=(10, 6))
df.boxplot(column='age', by='target', figsize=(10, 6))
plt.title('Age Distribution by Heart Disease Status')
plt.suptitle('')
plt.xlabel('Heart Disease (0: No, 1: Yes)')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
print("Age distribution saved as 'age_distribution.png'")

# ===== MODEL TRAINING =====
print("\n=== MODEL TRAINING ===")

# Select features for modeling
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[feature_cols]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Train Decision Tree
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Select best model
if lr_accuracy >= dt_accuracy:
    best_model = lr_model
    best_pred = lr_pred
    model_name = "Logistic Regression"
    print(f"\nBest Model: Logistic Regression")
else:
    best_model = dt_model
    best_pred = dt_pred
    model_name = "Decision Tree"
    print(f"\nBest Model: Decision Tree")

# ===== MODEL EVALUATION =====
print("\n=== MODEL EVALUATION ===")

# Classification Report
print(f"\nClassification Report for {model_name}:")
print(classification_report(y_test, best_pred, target_names=['No Disease', 'Disease']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")

# ROC Curve
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {model_name}')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print(f"ROC curve saved as 'roc_curve.png' (AUC = {roc_auc:.4f})")

# ===== FEATURE IMPORTANCE =====
print("\n=== FEATURE IMPORTANCE ===")

if model_name == "Logistic Regression":
    # For Logistic Regression, use coefficients
    feature_importance = np.abs(lr_model.coef_[0])
else:
    # For Decision Tree, use feature_importances_
    feature_importance = dt_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Feature Importance - {model_name}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'feature_importance.png'")

# ===== SAVE MODEL AND ARTIFACTS =====
print("\n=== SAVING MODEL ===")

joblib.dump(best_model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save feature names
with open('feature_names.txt', 'w') as f:
    f.write(','.join(feature_cols))

print("Model saved as 'heart_disease_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("Label encoders saved as 'label_encoders.pkl'")
print("Feature names saved as 'feature_names.txt'")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"\nFinal Model: {model_name}")
print(f"Accuracy: {accuracy_score(y_test, best_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nTop 5 Most Important Features:")
print(feature_importance_df.head())
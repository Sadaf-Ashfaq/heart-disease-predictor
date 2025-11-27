"""
Evaluate Model Predictions and Test Accuracy
This script helps you test the model with known cases
"""

import joblib
import numpy as np
import pandas as pd

# Load model
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Model loaded successfully\n")
except:
    print("✗ Error: Model files not found. Run train_model.py first.")
    exit()

# Test cases with known outcomes
test_cases = [
    {
        'name': 'High Risk Patient (Should be HIGH)',
        'features': [65, 1, 3, 160, 280, 1, 2, 110, 1, 3.0, 2, 3, 2],
        'expected': 'High Risk'
    },
    {
        'name': 'Low Risk Patient (Should be LOW)',
        'features': [35, 0, 0, 110, 180, 0, 0, 170, 0, 0.5, 0, 0, 0],
        'expected': 'Low Risk'
    },
    {
        'name': 'Moderate Risk Patient (Should be MODERATE)',
        'features': [55, 1, 1, 140, 220, 0, 1, 140, 0, 1.5, 1, 1, 1],
        'expected': 'Moderate Risk'
    },
    {
        'name': 'Elderly Low Risk (Should be LOW-MODERATE)',
        'features': [70, 0, 0, 120, 200, 0, 0, 160, 0, 1.0, 0, 0, 0],
        'expected': 'Low-Moderate Risk'
    },
    {
        'name': 'Young High Risk (Should be MODERATE-HIGH)',
        'features': [45, 1, 3, 150, 260, 1, 1, 120, 1, 2.5, 2, 2, 2],
        'expected': 'Moderate-High Risk'
    }
]

def get_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

print("="*70)
print("MODEL PREDICTION EVALUATION")
print("="*70)
print("\nTesting model with known cases...\n")

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{i}. {test_case['name']}")
    print("-" * 70)
    
    # Create feature array
    features = np.array(test_case['features']).reshape(1, -1)
    
    # Print feature values
    print("   Input Features:")
    for name, value in zip(feature_names, test_case['features']):
        print(f"      {name:12s}: {value}")
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    risk_level = get_risk_level(probability)
    
    # Display results
    print(f"\n   Prediction Results:")
    print(f"      Disease Detected: {'YES' if prediction == 1 else 'NO'}")
    print(f"      Probability: {probability*100:.1f}%")
    print(f"      Risk Level: {risk_level}")
    print(f"      Expected: {test_case['expected']}")
    
    # Check if prediction is reasonable
    if ('High' in test_case['expected'] and probability >= 0.6) or \
       ('Low' in test_case['expected'] and probability < 0.4) or \
       ('Moderate' in test_case['expected'] and 0.3 <= probability <= 0.7):
        print(f"      Status: ✓ REASONABLE")
    else:
        print(f"      Status: ⚠ MAY NEED REVIEW")

print("\n" + "="*70)

# Test with actual dataset if available
print("\n\nVALIDATING AGAINST TRAINING DATA")
print("="*70)

try:
    df = pd.read_csv('heart_disease.csv')
    print(f"✓ Dataset loaded: {len(df)} records")
    
    # Show disease distribution
    df['target'] = (df['num'] > 0).astype(int)
    disease_count = df['target'].sum()
    no_disease_count = len(df) - disease_count
    
    print(f"\nDataset Distribution:")
    print(f"   No Disease: {no_disease_count} ({no_disease_count/len(df)*100:.1f}%)")
    print(f"   Disease: {disease_count} ({disease_count/len(df)*100:.1f}%)")
    
    print(f"\nModel should predict around {disease_count/len(df)*100:.1f}% positive cases")
    print("If your predictions are very different, the model may need retraining")
    
except Exception as e:
    print(f"✗ Could not load dataset: {e}")

print("\n" + "="*70)
print("\nRECOMMENDATIONS:")
print("="*70)
print("""
1. If predictions seem inaccurate:
   - Retrain the model: python train_model.py
   - This will use improved algorithms (Random Forest, Gradient Boosting)
   - Will perform hyperparameter tuning for better accuracy

2. Check model_info.txt for current model statistics:
   - Accuracy should be > 0.75 (75%)
   - ROC-AUC should be > 0.80 (80%)
   - Cross-validation score should be consistent with accuracy

3. Review feature_importance.png to see which factors matter most

4. Test with your own data and compare with medical expectations

5. Remember: This is a prediction tool, not a diagnosis tool
   - Always consult healthcare professionals
   - Use predictions as a screening tool only
""")


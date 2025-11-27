"""
Debug Script to Check System Status
Run this to diagnose issues with your Heart Disease Prediction System
"""

import os
import sys

print("="*60)
print("HEART DISEASE PREDICTION SYSTEM - DIAGNOSTIC CHECK")
print("="*60)

# Check Python version
print(f"\n✓ Python Version: {sys.version}")

# Check if required files exist
print("\n📁 CHECKING FILES:")
required_files = [
    'train_model.py',
    'app.py',
    'requirements.txt',
    'heart_disease.csv'
]

model_files = [
    'heart_disease_model.pkl',
    'scaler.pkl',
    'label_encoders.pkl',
    'feature_names.txt'
]

template_files = [
    'templates/login.html',
    'templates/register.html',
    'templates/dashboard.html',
    'templates/history.html'
]

all_files = required_files + model_files + template_files

for file in all_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING!")

# Check if model is trained
print("\n🤖 MODEL STATUS:")
if all(os.path.exists(f) for f in model_files):
    print("   ✓ Model files found - System ready for predictions")
else:
    print("   ✗ Model files missing - Run 'python train_model.py' first")

# Check installed packages
print("\n📦 CHECKING INSTALLED PACKAGES:")
required_packages = [
    'flask',
    'pandas',
    'numpy',
    'sklearn',
    'matplotlib',
    'seaborn',
    'joblib',
    'flask_sqlalchemy',
    'flask_bcrypt',
    'flask_login'
]

for package in required_packages:
    try:
        __import__(package.replace('_', '.'))
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED!")

# Try to load the model
print("\n🔍 TESTING MODEL LOADING:")
try:
    import joblib
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("   ✓ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
except FileNotFoundError as e:
    print(f"   ✗ Model file not found: {e}")
    print("   → Run 'python train_model.py' to create model files")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")

# Check database
print("\n💾 DATABASE STATUS:")
if os.path.exists('heart_disease.db'):
    print("   ✓ Database file exists")
    try:
        import sqlite3
        conn = sqlite3.connect('heart_disease.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"   Tables: {', '.join([t[0] for t in tables])}")
        
        # Count users
        try:
            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            print(f"   Users registered: {user_count}")
        except:
            print("   User table not initialized yet")
        
        conn.close()
    except Exception as e:
        print(f"   ✗ Error reading database: {e}")
else:
    print("   ⚠ Database not created yet (will be created when app starts)")

# Test prediction function
print("\n🧪 TESTING PREDICTION FUNCTION:")
try:
    import numpy as np
    import joblib
    
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Test data
    test_features = np.array([[50, 1, 0, 120, 200, 0, 0, 150, 0, 1.0, 0, 0, 0]]).reshape(1, -1)
    test_scaled = scaler.transform(test_features)
    prediction = model.predict(test_scaled)[0]
    probability = model.predict_proba(test_scaled)[0][1]
    
    print(f"   ✓ Prediction successful")
    print(f"   Test result: {'Disease' if prediction == 1 else 'No Disease'}")
    print(f"   Probability: {probability*100:.1f}%")
except Exception as e:
    print(f"   ✗ Prediction test failed: {e}")

# Port check
print("\n🌐 NETWORK:")
print("   Default app URL: http://127.0.0.1:5000")
print("   Make sure no other application is using port 5000")

# Summary
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)

issues = []
if not all(os.path.exists(f) for f in required_files):
    issues.append("Missing required files")
if not all(os.path.exists(f) for f in model_files):
    issues.append("Model not trained")
if not all(os.path.exists(f) for f in template_files):
    issues.append("Missing template files")

if not issues:
    print("✓ ALL CHECKS PASSED!")
    print("\n🚀 You're ready to run: python app.py")
else:
    print("⚠ ISSUES FOUND:")
    for issue in issues:
        print(f"   • {issue}")
    print("\n📝 RECOMMENDED ACTIONS:")
    if "Model not trained" in issues:
        print("   1. Run: python train_model.py")
    if "Missing template files" in issues:
        print("   2. Create templates/ folder and add HTML files")
    if "Missing required files" in issues:
        print("   3. Make sure all Python files are in the project root")

print("\n" + "="*60)
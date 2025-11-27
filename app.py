from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_result = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Store input features
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    cp = db.Column(db.String(50))
    trestbps = db.Column(db.Integer)
    chol = db.Column(db.Integer)
    thalch = db.Column(db.Integer)

# Load ML model and artifacts
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    with open('feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split(',')
    model_loaded = True
except:
    model_loaded = False
    print("Warning: Model files not found. Please run train_model.py first.")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "success"
    elif probability < 0.6:
        return "Moderate Risk", "warning"
    else:
        return "High Risk", "danger"

def get_precautions(probability, input_data):
    """Generate personalized safety precautions"""
    precautions = []
    
    if probability >= 0.6:
        precautions.append("🚨 Consult a cardiologist immediately for a comprehensive heart evaluation")
        precautions.append("📊 Schedule regular ECG and stress tests")
        precautions.append("💊 Discuss medication options with your doctor")
    elif probability >= 0.3:
        precautions.append("👨‍⚕️ Schedule a check-up with your doctor within the next month")
        precautions.append("📈 Monitor your blood pressure and cholesterol levels regularly")
    else:
        precautions.append("✅ Maintain your current healthy lifestyle")
        precautions.append("🔍 Continue with annual health check-ups")
    
    # Safely get input values with default fallbacks
    try:
        age = int(input_data.get('age', 0))
        trestbps = int(input_data.get('trestbps', 0))
        chol = int(input_data.get('chol', 0))
        thalch = int(input_data.get('thalch', 0))
    except (ValueError, TypeError):
        age = 0
        trestbps = 0
        chol = 0
        thalch = 0
    
    if age > 55:
        precautions.append("👴 Age is a factor - maintain regular cardiovascular monitoring")
    if trestbps > 140:
        precautions.append("🩺 Your blood pressure is elevated - work on reducing it through diet, exercise, and medication if prescribed")
    if chol > 240:
        precautions.append("🥗 Your cholesterol is high - adopt a heart-healthy diet low in saturated fats")
    if thalch < 120:
        precautions.append("🏃 Your maximum heart rate is low - gradually increase physical activity with doctor's guidance")
    
    precautions.extend([
        "🥦 Follow a Mediterranean diet rich in fruits, vegetables, whole grains, and healthy fats",
        "🏋️ Aim for at least 150 minutes of moderate exercise per week",
        "🚭 If you smoke, seek help to quit immediately",
        "😴 Get 7-9 hours of quality sleep each night",
        "🧘 Practice stress management techniques like meditation or yoga",
        "⚖️ Maintain a healthy weight (BMI 18.5-24.9)",
        "🧂 Limit sodium intake to less than 2,300mg per day",
        "🍷 Limit alcohol consumption"
    ])
    
    return precautions

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'danger')
            return render_template('register.html')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return render_template('register.html')
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if not model_loaded:
        flash('Model not loaded. Please contact administrator.', 'danger')
    
    # Get user's recent predictions
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', predictions=recent_predictions)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded',
            'precautions': ['Please contact administrator - model files not found']
        }), 500
    
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'error': 'No data received',
                'precautions': ['Please fill in all fields']
            }), 400
        
        features = []
        input_data = {}
        
        for feature in feature_names:
            value = data.get(feature)
            input_data[feature] = value
            
            if feature in label_encoders:
                if feature == 'sex':
                    value = 1 if value.lower() == 'male' else 0
                elif feature == 'cp':
                    cp_mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}
                    value = cp_mapping.get(value.lower(), 0)
                elif feature == 'fbs':
                    value = 1 if value == 'true' or value == '1' or value == 1 else 0
                elif feature == 'restecg':
                    restecg_mapping = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
                    value = restecg_mapping.get(value.lower(), 0)
                elif feature == 'exang':
                    value = 1 if value == 'true' or value == '1' or value == 1 else 0
                elif feature == 'slope':
                    slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
                    value = slope_mapping.get(value.lower(), 0)
                elif feature == 'thal':
                    thal_mapping = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
                    value = thal_mapping.get(value.lower(), 0)
            
            features.append(float(value))
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        risk_level, risk_class = get_risk_level(probability)
        precautions = get_precautions(probability, input_data)
        
        # Save prediction to database
        new_prediction = Prediction(
            user_id=current_user.id,
            prediction_result=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            age=int(input_data.get('age', 0)),
            sex=str(input_data.get('sex', '')),
            cp=str(input_data.get('cp', '')),
            trestbps=int(input_data.get('trestbps', 0)),
            chol=int(input_data.get('chol', 0)),
            thalch=int(input_data.get('thalch', 0))
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_class': risk_class,
            'precautions': precautions,
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print("Error in prediction:")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'precautions': ['An error occurred. Please try again or contact support.']
        }), 400

@app.route('/history')
@login_required
def history():
    all_predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=all_predictions)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
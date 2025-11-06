#!/usr/bin/env python3
"""
Diabetes Risk Prediction Web Application
A simple web app that predicts diabetes risk using minimal user inputs
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['age', 'bmi', 'family_history', 'frequent_urination', 
                             'excessive_thirst', 'fatigue', 'blurred_vision']
        self.setup_model()
    
    def create_synthetic_data(self):
        """Create synthetic diabetes data for training"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        data = {
            'age': np.random.normal(45, 15, n_samples).clip(18, 80),
            'bmi': np.random.normal(26, 5, n_samples).clip(15, 50),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'frequent_urination': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'excessive_thirst': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'fatigue': np.random.choice([0, 1], n_samples, p=[0.55, 0.45]),
            'blurred_vision': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        }
        
        # Create target variable with realistic correlations
        diabetes_risk = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = 0
            
            # Age factor
            if data['age'][i] > 45:
                risk_score += 0.3
            if data['age'][i] > 60:
                risk_score += 0.2
                
            # BMI factor
            if data['bmi'][i] > 25:
                risk_score += 0.2
            if data['bmi'][i] > 30:
                risk_score += 0.3
                
            # Family history
            if data['family_history'][i]:
                risk_score += 0.4
                
            # Symptoms
            symptom_count = (data['frequent_urination'][i] + 
                           data['excessive_thirst'][i] + 
                           data['fatigue'][i] + 
                           data['blurred_vision'][i])
            risk_score += symptom_count * 0.15
            
            # Add some randomness
            risk_score += np.random.normal(0, 0.1)
            
            # Convert to binary
            diabetes_risk[i] = 1 if risk_score > 0.6 else 0
        
        data['diabetes'] = diabetes_risk.astype(int)
        return pd.DataFrame(data)
    
    def setup_model(self):
        """Train the diabetes prediction model"""
        print("Setting up diabetes prediction model...")
        
        # Create synthetic training data
        df = self.create_synthetic_data()
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['diabetes']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2%}")
        
        # Save model and scaler
        joblib.dump(self.model, 'diabetes_model.pkl')
        joblib.dump(self.scaler, 'diabetes_scaler.pkl')
        print("Model trained and saved successfully!")
    
    def predict_risk(self, user_input):
        """Predict diabetes risk for user input"""
        try:
            # Convert input to numpy array
            features = np.array([[
                user_input['age'],
                user_input['bmi'],
                user_input['family_history'],
                user_input['frequent_urination'],
                user_input['excessive_thirst'],
                user_input['fatigue'],
                user_input['blurred_vision']
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Calculate risk percentage
            risk_percentage = probability[1] * 100
            
            # Determine risk level
            if risk_percentage < 30:
                risk_level = "Low"
                risk_color = "success"
            elif risk_percentage < 60:
                risk_level = "Moderate"
                risk_color = "warning"
            else:
                risk_level = "High"
                risk_color = "danger"
            
            return {
                'prediction': int(prediction),
                'risk_percentage': round(risk_percentage, 1),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'recommendations': self.get_recommendations(risk_level, user_input)
            }
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_recommendations(self, risk_level, user_input):
        """Generate personalized recommendations"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.append("🏥 Consult a healthcare professional immediately for proper testing")
            recommendations.append("📊 Get HbA1c and fasting glucose tests done")
        
        if risk_level in ["High", "Moderate"]:
            recommendations.append("🥗 Adopt a balanced diet low in sugar and refined carbs")
            recommendations.append("🏃‍♂️ Engage in regular physical activity (150 min/week)")
            recommendations.append("⚖️ Monitor your weight and maintain a healthy BMI")
        
        if user_input['bmi'] > 25:
            recommendations.append("📉 Work on weight management with professional guidance")
        
        if any([user_input['frequent_urination'], user_input['excessive_thirst'], 
                user_input['fatigue'], user_input['blurred_vision']]):
            recommendations.append("🔍 Monitor symptoms and track any changes")
        
        if risk_level == "Low":
            recommendations.append("✅ Maintain your current healthy lifestyle")
            recommendations.append("🔄 Regular health check-ups are still important")
        
        recommendations.append("💧 Stay hydrated and get adequate sleep")
        recommendations.append("🚭 Avoid smoking and limit alcohol consumption")
        
        return recommendations

# Initialize Flask app and predictor
app = Flask(__name__)
app.secret_key = 'diabetes_predictor_secret_key_2024'
predictor = DiabetesPredictor()

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('diabetes_home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        user_input = {
            'age': float(request.form.get('age', 0)),
            'bmi': float(request.form.get('bmi', 0)),
            'family_history': int(request.form.get('family_history', 0)),
            'frequent_urination': int(request.form.get('frequent_urination', 0)),
            'excessive_thirst': int(request.form.get('excessive_thirst', 0)),
            'fatigue': int(request.form.get('fatigue', 0)),
            'blurred_vision': int(request.form.get('blurred_vision', 0))
        }
        
        # Validate input
        if user_input['age'] < 1 or user_input['age'] > 120:
            raise ValueError("Age must be between 1 and 120")
        if user_input['bmi'] < 10 or user_input['bmi'] > 60:
            raise ValueError("BMI must be between 10 and 60")
        
        # Get prediction
        result = predictor.predict_risk(user_input)
        
        if result:
            return render_template('diabetes_result.html', 
                                 user_input=user_input, 
                                 result=result)
        else:
            return render_template('diabetes_home.html', 
                                 error="Prediction failed. Please try again.")
    
    except ValueError as e:
        return render_template('diabetes_home.html', 
                             error=f"Invalid input: {e}")
    except Exception as e:
        return render_template('diabetes_home.html', 
                             error="An error occurred. Please check your input and try again.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = predictor.predict_risk(data)
        return jsonify(result) if result else jsonify({'error': 'Prediction failed'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    """About page with information about the model"""
    return render_template('diabetes_about.html')

if __name__ == '__main__':
    print("Starting Diabetes Risk Prediction Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
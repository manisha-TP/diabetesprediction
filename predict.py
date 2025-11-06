"""
Diabetes Prediction Script
This script allows making predictions on new data using a trained model.
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diabetes_detector import DiabetesDetector


def predict_single(detector, features):
    """
    Make a prediction for a single patient.
    
    Args:
        detector (DiabetesDetector): Trained diabetes detector
        features (dict or list): Patient features
        
    Returns:
        prediction (int): 0 for no diabetes, 1 for diabetes
        probability (float): Probability of having diabetes
    """
    if isinstance(features, dict):
        # Convert dict to list in correct order
        features = [features[name] for name in detector.feature_names]
    
    features_array = np.array(features).reshape(1, -1)
    features_scaled = detector.scaler.transform(features_array)
    
    prediction = detector.predict(features_scaled)[0]
    probability = detector.predict_proba(features_scaled)[0][1]
    
    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description='Predict diabetes for a patient')
    parser.add_argument('--model', type=str, default='models/diabetes_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--pregnancies', type=float, required=True,
                        help='Number of pregnancies')
    parser.add_argument('--glucose', type=float, required=True,
                        help='Plasma glucose concentration')
    parser.add_argument('--blood-pressure', type=float, required=True,
                        help='Diastolic blood pressure (mm Hg)')
    parser.add_argument('--skin-thickness', type=float, required=True,
                        help='Triceps skin fold thickness (mm)')
    parser.add_argument('--insulin', type=float, required=True,
                        help='2-Hour serum insulin (mu U/ml)')
    parser.add_argument('--bmi', type=float, required=True,
                        help='Body mass index (weight in kg/(height in m)^2)')
    parser.add_argument('--diabetes-pedigree', type=float, required=True,
                        help='Diabetes pedigree function')
    parser.add_argument('--age', type=float, required=True,
                        help='Age (years)')
    
    args = parser.parse_args()
    
    # Load model
    detector = DiabetesDetector()
    detector.load_model(args.model)
    
    # Prepare features
    features = [
        args.pregnancies,
        args.glucose,
        args.blood_pressure,
        args.skin_thickness,
        args.insulin,
        args.bmi,
        args.diabetes_pedigree,
        args.age
    ]
    
    # Make prediction
    prediction, probability = predict_single(detector, features)
    
    # Display results
    print("\n" + "="*50)
    print("DIABETES PREDICTION RESULTS")
    print("="*50)
    print(f"\nPatient Features:")
    print(f"  Pregnancies: {args.pregnancies}")
    print(f"  Glucose: {args.glucose}")
    print(f"  Blood Pressure: {args.blood_pressure}")
    print(f"  Skin Thickness: {args.skin_thickness}")
    print(f"  Insulin: {args.insulin}")
    print(f"  BMI: {args.bmi}")
    print(f"  Diabetes Pedigree Function: {args.diabetes_pedigree}")
    print(f"  Age: {args.age}")
    
    print(f"\n{'Prediction:':<30} {'DIABETES' if prediction == 1 else 'NO DIABETES'}")
    print(f"{'Probability of Diabetes:':<30} {probability:.2%}")
    print(f"{'Confidence:':<30} {max(probability, 1-probability):.2%}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

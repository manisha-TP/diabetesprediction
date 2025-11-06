"""
Train Diabetes Detection Model
This script trains a machine learning model for diabetes detection.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diabetes_detector import DiabetesDetector
from sklearn.model_selection import train_test_split


def train_model(data_path, model_type='logistic', test_size=0.2, save_path='models/diabetes_model.pkl'):
    """
    Train a diabetes detection model.
    
    Args:
        data_path (str): Path to the dataset CSV file
        model_type (str): Type of model ('logistic' or 'random_forest')
        test_size (float): Proportion of data to use for testing
        save_path (str): Path to save the trained model
    """
    print("="*60)
    print("DIABETES DETECTION MODEL TRAINING")
    print("="*60)
    
    # Initialize detector
    print(f"\nInitializing {model_type} model...")
    detector = DiabetesDetector(model_type=model_type)
    
    # Load data
    print(f"Loading data from {data_path}...")
    X, y = detector.load_data(data_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {', '.join(detector.feature_names)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    print(f"\nSplitting data (test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Preprocess
    print("\nPreprocessing data...")
    X_train_scaled, y_train = detector.preprocess_data(X_train, y_train, fit_scaler=True)
    X_test_scaled = detector.preprocess_data(X_test, fit_scaler=False)
    
    # Train
    print("Training model...")
    detector.train(X_train_scaled, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    train_metrics = detector.evaluate(X_train_scaled, y_train)
    print(f"\nTraining Accuracy: {train_metrics['accuracy']:.4f}")
    
    test_metrics = detector.evaluate(X_test_scaled, y_test)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(test_metrics['confusion_matrix'])
    
    print(f"\nClassification Report (Test Set):")
    print(test_metrics['classification_report'])
    
    # Save model
    print("="*60)
    detector.save_model(save_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return detector


def main():
    parser = argparse.ArgumentParser(description='Train diabetes detection model')
    parser.add_argument('--data', type=str, default='data/diabetes.csv',
                        help='Path to diabetes dataset CSV file')
    parser.add_argument('--model-type', type=str, default='logistic',
                        choices=['logistic', 'random_forest'],
                        help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--save-path', type=str, default='models/diabetes_model.pkl',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_type=args.model_type,
        test_size=args.test_size,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()

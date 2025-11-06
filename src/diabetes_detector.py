"""
Diabetes Detection Model
This module provides functionality for training and predicting diabetes using machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class DiabetesDetector:
    """
    A class for diabetes detection using machine learning models.
    """
    
    def __init__(self, model_type='logistic'):
        """
        Initialize the diabetes detector.
        
        Args:
            model_type (str): Type of model to use ('logistic' or 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'logistic' or 'random_forest'")
    
    def load_data(self, filepath):
        """
        Load diabetes dataset from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            X (DataFrame): Features
            y (Series): Target variable
        """
        df = pd.read_csv(filepath)
        
        # Assuming the last column is the target variable (Outcome)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def preprocess_data(self, X, y=None, fit_scaler=True):
        """
        Preprocess the data by scaling features.
        
        Args:
            X (DataFrame): Features
            y (Series): Target variable (optional)
            fit_scaler (bool): Whether to fit the scaler
            
        Returns:
            X_scaled (array): Scaled features
            y (Series): Target variable (if provided)
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def train(self, X_train, y_train):
        """
        Train the diabetes detection model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Predict diabetes for given features.
        
        Args:
            X (array): Features
            
        Returns:
            predictions (array): Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probability of diabetes for given features.
        
        Args:
            X (array): Features
            
        Returns:
            probabilities (array): Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
            
        Returns:
            metrics (dict): Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        return metrics
    
    def save_model(self, model_path='models/diabetes_model.pkl'):
        """
        Save the trained model and scaler.
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/diabetes_model.pkl'):
        """
        Load a trained model and scaler.
        
        Args:
            model_path (str): Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {model_path}")


def main():
    """
    Example usage of the DiabetesDetector class.
    """
    # Initialize detector
    detector = DiabetesDetector(model_type='logistic')
    
    # Load and preprocess data
    print("Loading data...")
    X, y = detector.load_data('data/sample_diabetes.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess
    print("Preprocessing data...")
    X_train_scaled, y_train = detector.preprocess_data(X_train, y_train, fit_scaler=True)
    X_test_scaled = detector.preprocess_data(X_test, fit_scaler=False)
    
    # Train
    print("Training model...")
    detector.train(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = detector.evaluate(X_test_scaled, y_test)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")
    
    # Save model
    detector.save_model()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

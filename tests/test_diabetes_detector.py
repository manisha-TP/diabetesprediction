"""
Unit tests for the Diabetes Detection System
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diabetes_detector import DiabetesDetector


class TestDiabetesDetector(unittest.TestCase):
    """Test cases for DiabetesDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = DiabetesDetector(model_type='logistic')
        
        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 10, 100),
            'Glucose': np.random.randint(80, 200, 100),
            'BloodPressure': np.random.randint(60, 100, 100),
            'SkinThickness': np.random.randint(10, 50, 100),
            'Insulin': np.random.randint(0, 300, 100),
            'BMI': np.random.uniform(18, 45, 100),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.0, 100),
            'Age': np.random.randint(21, 70, 100)
        })
        self.y = pd.Series(np.random.randint(0, 2, 100))
    
    def test_initialization(self):
        """Test detector initialization"""
        detector_lr = DiabetesDetector(model_type='logistic')
        self.assertIsNotNone(detector_lr.model)
        
        detector_rf = DiabetesDetector(model_type='random_forest')
        self.assertIsNotNone(detector_rf.model)
        
        with self.assertRaises(ValueError):
            DiabetesDetector(model_type='invalid')
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        X_scaled, y_scaled = self.detector.preprocess_data(self.X, self.y, fit_scaler=True)
        
        # Check that data is scaled
        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertIsNotNone(X_scaled)
        
        # Check that scaler was fitted
        self.assertIsNotNone(self.detector.scaler.mean_)
    
    def test_train_and_predict(self):
        """Test model training and prediction"""
        X_scaled, y_scaled = self.detector.preprocess_data(self.X, self.y, fit_scaler=True)
        
        # Train model
        self.detector.train(X_scaled, y_scaled)
        
        # Make predictions
        predictions = self.detector.predict(X_scaled)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_proba(self):
        """Test probability predictions"""
        X_scaled, y_scaled = self.detector.preprocess_data(self.X, self.y, fit_scaler=True)
        self.detector.train(X_scaled, y_scaled)
        
        probabilities = self.detector.predict_proba(X_scaled)
        
        self.assertEqual(probabilities.shape[0], len(self.y))
        self.assertEqual(probabilities.shape[1], 2)
        
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), 
            np.ones(len(self.y))
        )
    
    def test_evaluate(self):
        """Test model evaluation"""
        X_scaled, y_scaled = self.detector.preprocess_data(self.X, self.y, fit_scaler=True)
        self.detector.train(X_scaled, y_scaled)
        
        metrics = self.detector.evaluate(X_scaled, y_scaled)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        import tempfile
        
        X_scaled, y_scaled = self.detector.preprocess_data(self.X, self.y, fit_scaler=True)
        self.detector.train(X_scaled, y_scaled)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            self.detector.save_model(model_path)
            
            # Load model
            new_detector = DiabetesDetector()
            new_detector.load_model(model_path)
            
            # Check that loaded model makes same predictions
            pred1 = self.detector.predict(X_scaled)
            pred2 = new_detector.predict(X_scaled)
            
            np.testing.assert_array_equal(pred1, pred2)
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)


class TestDataFormat(unittest.TestCase):
    """Test data format requirements"""
    
    def test_feature_count(self):
        """Test that dataset has correct number of features"""
        detector = DiabetesDetector()
        
        # Sample data should have 8 features
        sample_data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'sample_diabetes.csv'
        )
        
        if os.path.exists(sample_data_path):
            X, y = detector.load_data(sample_data_path)
            self.assertEqual(X.shape[1], 8, "Dataset should have 8 features")


if __name__ == '__main__':
    unittest.main()

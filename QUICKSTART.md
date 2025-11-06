# Diabetes Detection - Quick Start Guide

This guide shows you how to quickly get started with the diabetes detection system.

## Installation

```bash
pip install -r requirements.txt
```

## Training a Model

Train a model using the provided sample dataset:

```bash
python train.py --data data/sample_diabetes.csv --model-type logistic
```

Output:
```
============================================================
DIABETES DETECTION MODEL TRAINING
============================================================

Initializing logistic model...
Loading data from data/sample_diabetes.csv...
Dataset shape: (99, 8)
Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Training set size: 79
Test set size: 20

Training model...
Test Accuracy: 0.7000

Model saved to models/diabetes_model.pkl
============================================================
```

## Making Predictions

### Example 1: High-Risk Patient

```bash
python predict.py \
  --pregnancies 6 \
  --glucose 148 \
  --blood-pressure 72 \
  --skin-thickness 35 \
  --insulin 0 \
  --bmi 33.6 \
  --diabetes-pedigree 0.627 \
  --age 50
```

Result: **DIABETES** (81.39% probability)

### Example 2: Low-Risk Patient

```bash
python predict.py \
  --pregnancies 1 \
  --glucose 85 \
  --blood-pressure 66 \
  --skin-thickness 29 \
  --insulin 0 \
  --bmi 26.6 \
  --diabetes-pedigree 0.351 \
  --age 31
```

Result: **NO DIABETES** (15.71% probability)

## Using the Python API

```python
from src.diabetes_detector import DiabetesDetector
import numpy as np

# Load a trained model
detector = DiabetesDetector()
detector.load_model('models/diabetes_model.pkl')

# Prepare patient data
patient_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale the data
patient_data_scaled = detector.scaler.transform(patient_data)

# Make prediction
prediction = detector.predict(patient_data_scaled)[0]
probability = detector.predict_proba(patient_data_scaled)[0]

print(f"Prediction: {'DIABETES' if prediction == 1 else 'NO DIABETES'}")
print(f"Probability: {probability[1]:.2%}")
```

## Running Tests

```bash
python -m unittest tests.test_diabetes_detector -v
```

All tests should pass:
```
test_feature_count ... ok
test_evaluate ... ok
test_initialization ... ok
test_predict_proba ... ok
test_preprocess_data ... ok
test_save_and_load_model ... ok
test_train_and_predict ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.034s

OK
```

## Model Features

The diabetes detection model uses 8 features:

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (mg/dL)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**: Genetic likelihood score
8. **Age**: Age in years

## Performance Metrics

On the sample dataset:
- Training Accuracy: ~73%
- Test Accuracy: ~70%
- Model Size: ~2KB

## Next Steps

1. **Get more data**: Download the full Pima Indians Diabetes Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. **Try different models**: Use `--model-type random_forest` for Random Forest
3. **Tune hyperparameters**: Modify the model parameters in `src/diabetes_detector.py`
4. **Add more features**: Extend the model with additional medical data
5. **Deploy**: Create a web API using Flask or FastAPI

## Troubleshooting

### Import Errors
Make sure you're running scripts from the project root directory:
```bash
cd /path/to/diabetesprediction
python train.py
```

### Missing Dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

### Model Not Found
Train a model first before making predictions:
```bash
python train.py --data data/sample_diabetes.csv
```

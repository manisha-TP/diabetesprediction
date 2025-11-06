# Diabetes Prediction System

A machine learning-based system for detecting and predicting diabetes using patient medical data.

## Overview

This project implements a diabetes detection system using machine learning algorithms. It can train models on diabetes datasets and make predictions for new patients based on their medical features.

## Features

- **Multiple ML Models**: Support for Logistic Regression and Random Forest classifiers
- **Data Preprocessing**: Automatic feature scaling and data preparation
- **Model Training**: Easy-to-use training pipeline with train/test split
- **Prediction**: Command-line interface for making predictions on new patient data
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Accuracy, confusion matrix, and classification reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/manisha-TP/diabetesprediction.git
cd diabetesprediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The system works with the Pima Indians Diabetes Dataset. The dataset should contain the following features:

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index
7. **DiabetesPedigreeFunction**: Diabetes pedigree function
8. **Age**: Age in years
9. **Outcome**: Target variable (0 = no diabetes, 1 = diabetes)

A sample dataset is provided in `data/sample_diabetes.csv` for testing. For full dataset, download from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Usage

### Training a Model

Train a diabetes detection model using your dataset:

```bash
python train.py --data data/sample_diabetes.csv --model-type logistic
```

Options:
- `--data`: Path to the CSV dataset (default: `data/diabetes.csv`)
- `--model-type`: Type of model - `logistic` or `random_forest` (default: `logistic`)
- `--test-size`: Proportion of data for testing (default: `0.2`)
- `--save-path`: Path to save the trained model (default: `models/diabetes_model.pkl`)

### Making Predictions

Predict diabetes for a new patient:

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

The script will output:
- Patient features
- Prediction (DIABETES or NO DIABETES)
- Probability of diabetes
- Confidence level

### Using as a Python Module

```python
from src.diabetes_detector import DiabetesDetector
from sklearn.model_selection import train_test_split

# Initialize detector
detector = DiabetesDetector(model_type='logistic')

# Load and prepare data
X, y = detector.load_data('data/sample_diabetes.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess and train
X_train_scaled, y_train = detector.preprocess_data(X_train, y_train, fit_scaler=True)
detector.train(X_train_scaled, y_train)

# Evaluate
X_test_scaled = detector.preprocess_data(X_test, fit_scaler=False)
metrics = detector.evaluate(X_test_scaled, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save model
detector.save_model('models/my_model.pkl')

# Make predictions
new_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
new_data_scaled = detector.scaler.transform(new_data)
prediction = detector.predict(new_data_scaled)
probability = detector.predict_proba(new_data_scaled)
```

## Project Structure

```
diabetesprediction/
├── src/
│   └── diabetes_detector.py    # Main DiabetesDetector class
├── data/
│   ├── README.md               # Dataset information
│   └── sample_diabetes.csv     # Sample dataset
├── models/                     # Directory for saved models
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Performance

When trained on the Pima Indians Diabetes Dataset, the models typically achieve:
- **Logistic Regression**: ~77-78% accuracy
- **Random Forest**: ~75-77% accuracy

Performance may vary based on:
- Data preprocessing
- Train/test split
- Hyperparameter tuning
- Dataset quality

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Joblib >= 1.0.0

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset: Pima Indians Diabetes Database from UCI Machine Learning Repository
- Inspired by various diabetes prediction research papers and implementations

# Diabetes Dataset

This directory contains the diabetes dataset for training and testing the diabetes detection model.

## Dataset Information

The Pima Indians Diabetes Dataset is a commonly used dataset for diabetes prediction. It contains medical data for female patients of Pima Indian heritage.

### Features

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
8. **Age**: Age (years)
9. **Outcome**: Class variable (0 or 1) - 1 indicates diabetes, 0 indicates no diabetes

## Getting the Dataset

The Pima Indians Diabetes Dataset can be obtained from:
- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/diabetes

Download the dataset and place it in this directory as `diabetes.csv`.

## Dataset Format

The CSV file should have the following format:

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
```

## Sample Data

A small sample dataset is provided in `sample_diabetes.csv` for testing purposes.

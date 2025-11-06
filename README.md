# Diabetes Risk Prediction Web App

A simple web application that predicts diabetes risk using machine learning based on minimal user inputs including age, BMI, family history, and symptoms.

## Features

🧠 **AI-Powered Prediction**: Uses Random Forest machine learning model with 90.5% accuracy
📊 **Risk Assessment**: Provides risk percentage and categorization (Low/Moderate/High)
💡 **Personalized Recommendations**: Offers tailored health advice based on risk factors
📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
🏥 **Educational Content**: Includes comprehensive diabetes information

## Technology Stack

- **Backend**: Python 3.11, Flask 2.3.3
- **Machine Learning**: scikit-learn, numpy, pandas
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Deployment**: Vercel-ready configuration

## Local Development

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python diabetes_predictor_app.py
```

4. Open your browser and go to: http://localhost:5000

## Project Structure

```
diabetes-predictor/
├── diabetes_predictor_app.py    # Main Flask application
├── requirements.txt             # Python dependencies
├── runtime.txt                 # Python version for deployment
├── vercel.json                 # Vercel deployment configuration
├── .gitignore                  # Git ignore file
├── api/
│   └── index.py               # Vercel serverless entry point
├── templates/
│   ├── diabetes_base.html     # Base template
│   ├── diabetes_home.html     # Home page with input form
│   ├── diabetes_result.html   # Results page
│   └── diabetes_about.html    # About page
├── diabetes_model.pkl         # Trained ML model (auto-generated)
├── diabetes_scaler.pkl        # Feature scaler (auto-generated)
└── deploy_test.py             # Deployment readiness checker
```

## Deployment

### Option 1: Vercel (Recommended)

1. **GitHub Method**:
   - Upload project to GitHub
   - Connect your GitHub repo to Vercel
   - Deploy automatically

2. **Vercel CLI Method**:
   ```bash
   npm i -g vercel
   vercel --prod
   ```

### Option 2: Other Platforms

The app is compatible with:
- **Heroku**: Add Procfile with `web: python diabetes_predictor_app.py`
- **Railway**: Direct deployment with automatic detection
- **PythonAnywhere**: Upload files and configure WSGI
- **DigitalOcean App Platform**: Use runtime.txt and requirements.txt

## Health Disclaimer

⚠️ **Important**: This application is for educational purposes only and should not replace professional medical advice. Always consult qualified healthcare providers for medical concerns.

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: Age, BMI, Family History, Symptoms (4 types)
- **Training Data**: Synthetic dataset based on medical research
- **Accuracy**: 90.5% on validation set
- **Risk Levels**: Low (<30%), Moderate (30-60%), High (>60%)

## Input Requirements

- **Age**: 1-120 years
- **BMI**: 10-60 kg/m²
- **Family History**: Yes/No for diabetes in immediate family
- **Symptoms**: Checkboxes for:
  - Frequent urination
  - Excessive thirst
  - Unusual fatigue
  - Blurred vision

## Features in Detail

### 🏠 Home Page
- User-friendly input form
- Built-in BMI calculator
- Input validation
- Responsive design

### 📊 Results Page
- Risk percentage and level
- Risk factor breakdown
- Personalized recommendations
- Visual risk assessment

### ℹ️ About Page
- Diabetes education
- Model explanation
- Risk factor information
- Prevention strategies

## Development Notes

- Flask runs in debug mode for development
- Model trains on first run and saves to disk
- Templates use Bootstrap 5 for styling
- All user inputs are validated
- Error handling implemented

## License

This project is for educational use. Model and recommendations should not be used for actual medical decisions.

## Support

For issues or questions about deployment, check the deployment test results:
```bash
python deploy_test.py
```
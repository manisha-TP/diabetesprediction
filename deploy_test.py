#!/usr/bin/env python3
"""
Deployment readiness checker for Diabetes Risk Prediction App
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imports successfully")
        return True
    except ImportError as e:
        print(f"❌ {module_name} import failed: {e}")
        return False

def check_templates():
    """Check if all required templates exist"""
    templates_dir = "templates"
    required_templates = [
        "diabetes_base.html",
        "diabetes_home.html", 
        "diabetes_result.html",
        "diabetes_about.html"
    ]
    
    all_good = True
    for template in required_templates:
        filepath = os.path.join(templates_dir, template)
        if not check_file_exists(filepath, f"Template {template}"):
            all_good = False
    
    return all_good

def main():
    print("🔍 Checking Diabetes Risk Prediction App Deployment Readiness...\n")
    
    all_checks_passed = True
    
    # Check main app file
    if not check_file_exists("diabetes_predictor_app.py", "Main app file"):
        all_checks_passed = False
    
    # Check deployment files
    deployment_files = [
        ("vercel.json", "Vercel configuration"),
        ("requirements.txt", "Python dependencies"),
        ("runtime.txt", "Python runtime version"),
        ("api/index.py", "Vercel entry point")
    ]
    
    for filepath, description in deployment_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check templates
    if not check_templates():
        all_checks_passed = False
    
    # Check Python imports
    print("\n📦 Checking Python dependencies...")
    required_modules = [
        "flask",
        "numpy", 
        "pandas",
        "sklearn",
        "joblib"
    ]
    
    for module in required_modules:
        if not check_import(module):
            all_checks_passed = False
    
    # Try importing the main app
    print("\n🔧 Testing Flask app import...")
    try:
        from diabetes_predictor_app import app
        print("✅ Flask app imports successfully")
        
        # Check if model files exist
        model_files = ["diabetes_model.pkl", "diabetes_scaler.pkl"]
        for model_file in model_files:
            if not check_file_exists(model_file, f"Model file {model_file}"):
                print("ℹ️  Model files will be created on first run")
        
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        all_checks_passed = False
    
    print("\n" + "="*50)
    if all_checks_passed:
        print("🚀 Your app is ready for deployment!")
        print("\nDeployment options:")
        print("1. Upload to GitHub and connect with Vercel")
        print("2. Use Vercel CLI: vercel --prod")
        print("3. Deploy to other platforms (Heroku, Railway, etc.)")
    else:
        print("❌ Some issues need to be fixed before deployment")
    
    print("="*50)

if __name__ == "__main__":
    main()
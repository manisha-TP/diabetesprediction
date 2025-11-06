#!/usr/bin/env python3
"""
Vercel serverless function entry point for Diabetes Risk Prediction App
"""

import os
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diabetes_predictor_app import app

# For Vercel, we need to export the Flask app as 'app'
if __name__ == "__main__":
    app.run()
else:
    # This is the entry point for Vercel
    pass
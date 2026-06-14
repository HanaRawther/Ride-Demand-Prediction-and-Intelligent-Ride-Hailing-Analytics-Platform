Ride Demand Prediction System
Overview:

This project is a machine learning-based ride demand prediction system designed for ride-hailing platforms. It forecasts ride demand, estimates 
wait times, and supports surge pricing decisions using historical data from New York City and Bangalore.

Features:
Ride demand forecasting using machine learning models
Exploratory Data Analysis (EDA) with insights and visualizations
Feature engineering for time-based and location-based patterns
Transfer learning across cities with limited data
Surge pricing estimation module
Wait-time prediction system
Interactive dashboards for different user roles

Dataset:
New York City ride demand dataset
Bangalore ride demand dataset
Data processed for temporal, spatial, and demand-related features
Machine Learning Models
Multiple models evaluated and compared
Gradient Boosting selected as the final model due to best performance
Evaluation based on prediction error and generalization ability
Transfer Learning

A model trained on one city was adapted to another city with limited data, improving scalability and reducing the need for large datasets in new locations.

Web Application:
Frontend built using Streamlit
Backend developed using FastAPI
Interactive dashboards for:
Riders
Drivers
Administrators
Developers

Tech Stack:
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
FastAPI

How to Run:
pip install -r requirements.txt
streamlit run app.py

Project Goal:

To optimize ride-hailing operations by predicting demand patterns and enabling data-driven decisions for pricing, allocation, and user experience improvement.

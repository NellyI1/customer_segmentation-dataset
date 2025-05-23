# customer_segmentation-dataset
Clean and transformed dataset for Bosch price-optimization model project
# Bosch Customer Segmentation
This repository contains the exploratory data analysis (EDA) notebook and cleaned dataset for the Bosch price optimization using customer segmentation dataset business analytics project.

## Project Overview

This project focuses on preparing a clean, ready-to-analyze dataset that will be used for modeling customer segments and price optimization strategies.

## Dataset Description

Source: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation?resource=download)

##  Tasks Performed
Data was imported and previewed.
Data cleaning was carried out by identifying and handling missing values
Encoded categorical variables and numerical features were scaled.
Created visualizations (heatmap, segment distribution) to understand the correlation between several dataset components.
Feature engineering that introduces new features like Age group, Total family was done. Created 'Work_Exp_Age_Interaction' from 'Work_Experience' and 'Age' (scaled)
The cleaned dataset was saved for modeling.

## Files uploaded in Repository
customer segmentation dataset - Test.csv
optimization.py`: Python script for cleaning and preprocessing
optimization.ipynb: Jupyter script
`cleaned_customer_segmentation.csv`: Final cleaned dataset

## Tools Used

Python, Pandas, Seaborn & Matplotlib, Scikit-learn

## Next Steps
Perform customer clustering
Build predictive pricing models
Develop dashboards and reporting insights


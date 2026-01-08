# Linear vs. Tree-Based Models for Online News Popularity

## Overview
This project compares linear and non-linear models for predicting online news popularity using the *Online News Popularity* dataset.

Due to the heavy-tailed and noisy nature of the target variable, **Mean Absolute Error (MAE)** is used as the primary evaluation metric.

**Note**

This project serves as an exploratory study comparing linear and tree-based models.  
Further investigation revealed that self-reference share statistics introduce target leakage, which can inflate model performance.  
This insight motivated a revised evaluation strategy in subsequent work, focusing on leakage-free features and robust validation.

## Methods
- Exploratory Data Analysis (EDA)
- Ordinary Least Squares (OLS) regression with diagnostic analysis
- Random Forest regression as a non-linear alternative

Model diagnostics reveal substantial violations of linear model assumptions.  
A Random Forest model achieves improved explanatory power (**R² = 0.29**) with modest improvements in MAE.

## Data
The dataset is provided by Kaggle.  
Due to licensing restrictions, raw data is **not included** in this repository.

You can download the dataset from:  
https://www.kaggle.com/datasets/srikaranelakurthy/online-news-popularity

## Tech Stack
- **Language:** Python  
- **Libraries:** pandas, scikit-learn, matplotlib

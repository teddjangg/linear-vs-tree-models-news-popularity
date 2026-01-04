# linear-vs-tree-models-news-popularity

## Data
The dataset is from the UCI Machine Learning Repository / Kaggle.
Due to licensing restrictions, raw data is not included in this repository.

You can download the data from:
- https://www.kaggle.com/datasets/srikaranelakurthy/online-news-popularity

📄 **Open HTML report:**  
[linear-vs-tree-models-news-popularity.html](linear-vs-tree-models-news-popularity.html)

Comparing linear and non-linear models for predicting online news popularity.

This project explores the predictive performance of linear and non-linear models on the Online News Popularity dataset. Due to the heavy-tailed and
noisy nature of the target variable, Mean Absolute Error (MAE) is used as the primary evaluation metric.

Linear models such as Ordinary Least Squares (OLS) are first examined, followed by diagnostic analyses highlighting violations of model
assumptions. A Random Forest model is then applied as a non-linear alternative, achieving improved explanatory power (R² = 0.29) despite
modest gains in MAE.

**Techniques:** EDA, OLS, Random Forest, residual diagnostics  
**Language:** Python (pandas, scikit-learn, matplotlib)

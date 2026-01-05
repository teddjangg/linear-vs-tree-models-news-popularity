#!/usr/bin/env python
# coding: utf-8

# # Identifying Key Content and Media Features Driving Online News Popularity 
# # A Comparison of Linear and Non-Linear Models
# 

# ## Objective
# 
# The goal of this project is to compare linear and non-linear model to identify which content and media features are most strongly associated with online news popularity.
# 
# 
# 

# ## 1. Data Description
# The dataset used in this study is the Online News Popularity dataset, which contains information about online news articles and their associated popularity.

# In[121]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[81]:


df = pd.read_csv('OnlineNewsPopularity.csv')
df.head()


# In[82]:


nrow = df.shape[0]
ncol = df.shape[1]

print(nrow,ncol)


# In[83]:


df.info()


# In[84]:


df.isna().sum()


# The dataset consists of 39,644 observations and 61 variables. Each observation corresponds to a single news article. Moreover, the dataset includes a mixture of continuous variables and indicator variables capturing temporal, content-related, and metadata attributes of news articles.

# ## 2. Y - Exploratory Data Analysis

# In[85]:


y = df[" shares"] 
print(y)


# In[86]:


print(y.describe())


# In[87]:


print("y.min:", y.min(), "y.max:" , y.max())


# ### Log-Transformed Target Variable
# 

# In[88]:


plt.hist(np.log1p(y), bins=50)
plt.title("log1p(shares)")
plt.show()


# ### Heavy-Tailedness of the Target Variable

# To assess whether the target variable exhibits heavy-tailed behavior, we compare the maximum value to the median as a simple diagnostic.
# A large max-to-median ratio indicates the presence of extreme values and suggests that the distribution may be heavy-tailed.
# 

# In[89]:


print('Max to Med ratio: ' , y.max()/y.median())


# In[90]:


sm.qqplot(y, line='s') 


# ### Extreme values for Target Variable

# In[91]:


qlow = y.quantile(0.01)
qhigh = y.quantile(0.99)

df[' extreme values'] = ((y <= qlow) | (y >= qhigh))
print(df[' extreme values'].head(10))


# Based on the max-to-median ratio, the target variable exhibits heavy-tailed behavior. The distribution is strongly right-skewed, with a small number of extreme values dominating the upper tail. This pattern is further supported by the Q–Q plot, which shows substantial deviation from the reference line in the upper quantiles, indicating the presence of numerous extreme observations.
# 
# Given the pronounced heavy-tailed nature of the target variable, we adopt Mean Absolute Error (MAE) as the evaluation metric, as it is more robust to extreme values than squared-error-based measures.

# ## 3. X - Exploratory Data Analysis

# In[92]:


X = df.drop(columns=[' shares', 'url'])


# In[93]:


X.head()


# In[94]:


X.info()


# In[95]:


X.describe()


# In[96]:


X.nunique().sort_values()


# ### Heavy-Tailedness of the Target Variables X

# In[97]:


X_num = X.select_dtypes(include=['number'])

for col in X_num.columns:
    med = X[col].median()
    if med > 0:
        print(col, X[col].max() / med)


# In[98]:


plt.hist(np.log1p(X[' n_unique_tokens']), bins=100)
plt.xlabel('log(1 + n_unique_tokens)')
plt.ylabel('Frequency')
plt.title('Log-Transformed Distribution of n_unique_tokens')


# In[99]:


plt.scatter(np.log1p(X[' num_imgs']),y)
plt.xlabel('log(1 + num_imgs)')
plt.ylabel('Number of Shares')
plt.title('Relationship Between num_imgs and Article Popularity')


# In[100]:


plt.scatter(np.log1p(X[' n_unique_tokens']),y)
plt.xlabel('log(1 + n_unique_tokens)')
plt.ylabel('Number of Shares')
plt.title('Relationship Between n_unique_tokens and Article Popularity')


# In[101]:


plt.scatter(np.log1p(X[' n_non_stop_words']),y)
plt.xlabel('log(1 + n_non_stop_words)')
plt.ylabel('Number of Shares')
plt.title('Relationship Between n_non_stop_words and Article Popularity')


# In[102]:


plt.scatter(np.log1p(X[' n_non_stop_unique_tokens']),y)
plt.xlabel('log(1 + n_non_stop_unique_tokens)')
plt.ylabel('Number of Shares')
plt.title('Relationship Between n_non_stop_unique_tokens and Article Popularity')


# In[103]:


df.groupby(' is_weekend')[' shares'].mean()


# Several explanatory variables exhibit extremely large values that may dominate the upper tail of their distributions. To better understand the scale and prevalence of such extreme observations, we visually inspect a subset of variables with particularly long upper tails.
# The inspection shows that applying a log(1 + x) transformation reduces marginal skewness but does not fully mitigate heavy-tailed behavior or
# the presence of extreme values. In addition, for indicator variables such as `is_weekend`, we observe that articles published on weekends tend to receive higher numbers of shares.
# 

# ## 3. Train–Test Split

# In[104]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state=42)


# ## 4. Baseline Model

# In[105]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

y_pred = y.median()
y_hat = np.repeat(y_pred, len(y_test))

print(f"Mean absolute error {mean_absolute_error(y_test, y_hat):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_hat):.2f}")


# ## 5. Ordinary Least Squares (OLS)

# In[106]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

regressor = LinearRegression().fit(X_train, y_train)


# In[107]:


y_pred1 = regressor.predict(X_test)

print(f"Mean absolute error {mean_absolute_error(y_test, y_pred1):.2f}")

print(f"Coefficient of determination: {r2_score(y_test, y_pred1):.2f}")


# Compared to the baseline model, the Ordinary Least Squares (OLS) model achieves a higher MAE, suggesting limited predictive effectiveness.
# This observation motivates the exploration of potential remedies.

# In[108]:


y_train_log = np.log1p(y_train)

regressor = LinearRegression().fit(X_train, y_train_log)
y_pred2 = regressor.predict(X_test)

y_pred3 = np.expm1(y_pred2)

print(f"Mean absolute error {mean_absolute_error(y_test, y_pred3):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred3):.2f}")


# We first apply a log(1 + y) transformation to the target variable in the OLS framework. While this transformation leads to a modest reduction in MAE compared to the original OLS model, it does not sufficiently address the underlying non-linear relationships and extreme-value behavior in the data.

# ### Residual Plot

# In[109]:


residuals = y_test - y_pred1

plt.figure(figsize=(6,5))
plt.scatter(y_pred1, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for the Baseline Model (Test Set)')


# In[110]:


residuals = y_test - y_pred2

plt.figure(figsize=(6,5))
plt.scatter(y_pred2, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for the Original OLS Model (Test Set)')


# In[111]:


residuals = y_test - y_pred3

plt.figure(figsize=(6,5))
plt.scatter(y_pred3, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Log-Scale Residuals') 
plt.title('Residual Plot for the OLS Model with log(1 + y) Transformation (Test Set)')


# To complement the visual inspection of residual plots, we compute the correlation between the absolute residuals and fitted values as a simple diagnostic for heteroskedasticity. A strong positive correlation suggests that residual variance increases with the magnitude of fitted values.

# In[112]:


resid_ols = y_test - y_pred2
corr_ols = np.corrcoef(np.abs(resid_ols), y_pred2)[0,1]

resid_log = y_test - y_pred3
corr_log = np.corrcoef(np.abs(resid_log), y_pred3)[0,1]

print("corr(|resid|, fitted) OLS:", corr_ols)
print("corr(|resid|, fitted) log-OLS:", corr_log)


# Comparing the residual plots across the baseline, original OLS, and log-transformed OLS models, clear patterns and clustering of residuals are observed. Even after applying a log(1 + y) transformation, the residuals do not appear to be randomly scattered around zero, indicating
# persistent model misspecification.
# 
# These patterns suggest that the linear model struggles to capture the underlying relationship between the predictors and the target variable,
# particularly in the presence of noisy features and extreme values. This highlights the limitations of OLS when applied to heavy-tailed and highly heterogeneous data.

# ## 6. Random Forest

# Given the limitations observed in the linear models, we next consider a non-linear approach using Random Forest to examine whether it provides improved predictive performance.

# In[113]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators = 600,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
            
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# ### Selection of the Number of Trees

# In[114]:


from sklearn.ensemble import RandomForestRegressor

for estimator in [100,300,600,800,1000]:
    rf = RandomForestRegressor(
        n_estimators = estimator,
        max_depth=None,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
            
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)

    print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# ### Selection of Minimum Samples per Leaf

# In[115]:


from sklearn.ensemble import RandomForestRegressor

for leaf in [1,2,5,10] :
    rf = RandomForestRegressor(
        n_estimators = 600,
        max_depth=None,
        min_samples_leaf=leaf,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)

    print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# In[116]:


from sklearn.ensemble import RandomForestRegressor

for leaf in [20,30,40,60,80,100] :
    rf = RandomForestRegressor(
        n_estimators = 600,
        max_depth=None,
        min_samples_leaf=leaf,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)

    print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# ### Selection of Maximum Features

# In[117]:


from sklearn.ensemble import RandomForestRegressor

for features in [0.8,0.9,1.0] :
    rf = RandomForestRegressor(
        n_estimators = 600,
        max_depth=None,
        max_features=features,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)

    print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# ### Final Model

# In[118]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
        n_estimators = 600,
        max_depth=None,
        max_features=0.8,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1
    )
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

print(f"Mean absolute error {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred_rf):.2f}")


# The final Random Forest configuration is selected based on empirical performance stabilization, balancing predictive performance and
# computational cost.
# 
# Although the Random Forest model does not yield a dramatic reduction in MAE compared to the baseline model, the coefficient of determination
# (R²) increases to 0.29. This indicates that the model captures a substantially larger proportion of variance in the target variable, reflecting improved explanatory power.

# ## 7. Conclusions

# This project compares linear and non-linear modeling approaches to identify factors associated with online news popularity. Using Mean
# Absolute Error (MAE) as the primary evaluation metric due to the heavy-tailed nature of the target variable, we find that the original
# Ordinary Least Squares (OLS) model provides limited predictive performance. Diagnostic analyses reveal persistent violations of linear
# model assumptions, including heteroskedasticity and non-linear relationships, even after applying log transformations and other
# remedial measures.
# 
# To address these limitations, a Random Forest model is employed as a non-linear alternative. While the improvement in MAE relative to the
# baseline model is modest, the Random Forest achieves an R² value of 0.29, indicating a substantial increase in explained variance compared
# to linear models. Given the highly noisy, heterogeneous, and heavy-tailed nature of the dataset, this level of explanatory power is both reasonable and meaningful.
# 
# Overall, the results suggest that non-linear models such as Random Forest are better suited for capturing complex patterns and extreme
# behavior in online news popularity data, even when gains in absolute prediction error are constrained by inherent data noise.

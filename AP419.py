#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import statsmodels.api as sm


# In[2]:


# %matplotlib inline

# Load SAT dataset
sat_data = pd.read_csv('CASE1201.ASC.txt', delim_whitespace=True)


# In[3]:


print(sat_data.head()) # Print first 5 rows of the dataset


# In[4]:


print(sat_data.describe()) # Get basic statistics of numerical columns


# In[5]:


# Data visualization
sns.set(style="ticks", color_codes=True) # Set seaborn style
sns.pairplot(sat_data) # Plot pairwise relationships of columns
plt.show() # Show the plot


# In[6]:


plt.figure(figsize=(15,8))
plt.bar(sat_data['state'], sat_data['takers'])
plt.title('Number of SAT Takers per State')
plt.xlabel('State')
plt.ylabel('Number of SAT Takers')
plt.xticks(rotation=90)
plt.show()


# In[7]:


plt.figure(figsize=(8,5))
plt.hist(sat_data['sat'], bins=30, color='purple', alpha=0.75)
plt.title('Distribution of SAT Scores')
plt.xlabel('SAT Score')
plt.ylabel('Frequency')
plt.show()


# In[8]:


plt.figure(figsize=(6,6))
plt.pie(sat_data['takers'].value_counts(), labels=sat_data['takers'].value_counts().index, autopct='%1.1f%%', colors=['pink', 'lightblue'])
plt.title('Gender of SAT Takers')
plt.show()


# In[9]:


plt.figure(figsize=(8,5))
plt.scatter(sat_data['income'], sat_data['sat'], color='orange')
plt.title('Relationship between Income and SAT Scores')
plt.xlabel('Income')
plt.ylabel('SAT Score')
plt.show()


# In[10]:


plt.figure(figsize=(8,5))
plt.plot(sat_data['years'], sat_data['sat'], color='green', linewidth=2, marker='o', markersize=8)
plt.title('Trend in SAT Scores over the Years')
plt.xlabel('Year')
plt.ylabel('SAT Score')
plt.show()


# In[11]:


# Split into features (takers) and target (total scores)
X = sat_data.iloc[:, 2:]
y = sat_data.iloc[:, 1]


# In[12]:


# Simple linear regression model (intercept + takers)
X_sm = sm.add_constant(X)
X_np = X_sm.to_numpy()
sm_model = sm.OLS(y, X_np).fit()
lin_reg = LinearRegression()
lin_reg.fit(X.iloc[:, 1].to_numpy().reshape(-1, 1), y)
r2_simple = lin_reg.score(X.iloc[:, 1].to_numpy().reshape(-1, 1), y)
print(sm_model.summary())


# In[13]:


#variance of the model
sm_model.mse_resid


# In[14]:


# Confidence intervals
print(sm_model.conf_int(alpha=0.05))  # 95% confidence interval
print(sm_model.conf_int(alpha=0.01))  # 99% confidence interval


# In[15]:


# Full linear regression model
lin_model = sm.OLS(y,X).fit()
lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_full = lin_reg.score(X, y)
print(lin_model.summary())


# In[16]:


#variance of the model
lin_model.mse_resid


# In[17]:


# Confidence intervals
print(lin_model.conf_int(alpha=0.05))  # 95% confidence interval
print(lin_model.conf_int(alpha=0.01))  # 99% confidence interval


# #### ANOVA for model selection

# In[18]:


"""# Control Regaression"""

control_model = sm.OLS(y, sm.add_constant(X)).fit()
print(control_model.summary())

anova_lm(lin_model,control_model)

sns.residplot(y = 'sat',x = 'takers', data = sat_data)

sns.residplot(y = 'sat',x = 'income', data = sat_data);

fig = sm.graphics.qqplot(lin_model.resid, line='s')
plt.show()


# In[19]:


corr_matrix = X.corr()
print(corr_matrix)


# In[20]:


# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
lin_reg_pca = LinearRegression()
lin_reg_pca.fit(X_pca, y)
r2_pca = lin_reg_pca.score(X_pca, y)


# In[21]:


# Calculate F1 and P(F1) for each model
f1_simple, pval_simple = f_regression(X.iloc[:, 1].to_numpy().reshape(-1, 1), y)
f1_full, pval_full = f_regression(X, y)
f1_pca, pval_pca = f_regression(X_pca, y)


# In[22]:


print("Simple Linear Regression Model:")
print(f"R^2: {r2_simple:.3f}")
print(f"F1: {f1_simple[0]:.3f}")
print(f"P(F1): {pval_simple[0]:.3f}")

print("Full Linear Regression Model:")
print(f"R^2: {r2_full:.3f}")
print(f"F1: {f1_full[0]:.3f}")
print(f"P(F1): {pval_full[0]:.3f}")

print("PCA Model:")
print(f"R^2: {r2_pca:.3f}")
print(f"F1: {f1_pca[0]:.3f}")
print(f"P(F1): {pval_pca[0]:.3f}")


# In[23]:


# Simple linear regression model (intercept + takers) with PCA
X_sm_pca = sm.add_constant(X_pca)
sm_model_pca = sm.OLS(y, X_sm_pca).fit()
print(sm_model_pca.summary())


# In[24]:


# Full linear regression model with PCA
lin_model_pca = sm.OLS(y,X_pca).fit()
print(lin_model_pca.summary())


# In[25]:


# Feature selection using filter method (SelectKBest)
selector = SelectKBest(f_regression, k=2)
X_new = selector.fit_transform(X, y)
selected_features_filter = np.array(X.columns[selector.get_support()])
print(selected_features_filter)


# In[26]:


# Feature selection using wrapper method (RFE)
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=2, step=1)
selector = selector.fit(X, y)
selected_features_wrapper = np.array(X.columns[selector.support_])
print(selected_features_wrapper)


# In[27]:


# Feature selection using embedded method (SelectFromModel)
estimator = LinearRegression()
selector = SelectFromModel(estimator)
selector = selector.fit(X, y)
selected_features_embedded = np.array(X.columns[selector.get_support()])
print(selected_features_embedded)


# In[28]:


# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA components
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


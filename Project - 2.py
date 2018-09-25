
# coding: utf-8

# In[1]:


# Predicting players rating:


# In[2]:


# In this project you are going to predict the overall rating of soccer player based on their attributes such as 
# 'crossing', 'finishing etc.

# The dataset you are going to use is from European Soccer Database(https://www.kaggle.com/hugomathien/soccer)
# has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016.


# In[3]:


# About the Dataset:

# The ultimate Soccer database for data analysis and machine learning The dataset comes in the form of an SQL database and 
# contains statistics of about 25,000 football matches.
#from the top football league of 11 European Countries. It covers seasons from 2008 to 2016 and contains match statistics
#(i.e: scores, corners, fouls etc...) as well as the team formations, with player names and a pair of coordinates to indicate 
#their position on the pitch. +25,000 matches +10,000 players 11 European Countries with their lead championship Seasons 
#2008 to 2016 Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates 
#Team line up with squad formation (X, Y coordinates) Betting odds from up to 10 providers Detailed match events 
#(goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches The dataset also has a set of about 35 
#statistics for each player, derived from EA Sports' FIFA video games. It is not just the stats that come with a new version 
#of the game but also the weekly updates. So for instance if a player has performed poorly over a period of time and his stats 
#get impacted in FIFA, you would normally see the same in the dataset.


# In[4]:


# Importing libraries to the environment


# In[5]:


import sqlite3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[6]:


# Create connection.

cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df = df.dropna()


# In[11]:


df.info()


# In[12]:


# Feature Selection:

# Given that we have about 35-40 different features to play around with, we can attempt to run some feature selection algorithms
# to reduce the size of our featureset.


# In[13]:


from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE


# In[14]:


df_new = df.copy()


# In[15]:


# The most of the data is numeric. but there are a few objects and floating data types. In the subsequent prediction analysis 
# we’ll only concern ourself with the integer numerics, but there is obviously potential gains to be made by incorporating 
# the qualitative data (i.e. player position).


# In[16]:


df_new = df_new.select_dtypes(["int64", "float64"])


# In[17]:


df_new.shape


# In[18]:


df_new.info()


# In[19]:


x = df_new.drop('overall_rating', axis=1).values
y = df_new['overall_rating'].values.ravel()
from sklearn.preprocessing import scale
x = scale(x)


# In[20]:


df_1 = df_new.drop('overall_rating', axis=1)


# In[21]:


x.shape


# In[22]:


y.shape


# In[23]:


# Feature Selection using RFE Scikit Library:


# In[24]:


lm = LinearRegression()

rfe = RFE(lm, n_features_to_select = 10)
rfe_fit = rfe.fit(x,y)
features = []
for feat in df_1.columns[rfe_fit.support_]:
    print(feat)
    features.append(feat)


# In[25]:


features


# In[26]:


# Using Statsmodel to illustrate the summary results:


# In[27]:


lm = LinearRegression()

rfe = RFE(lm, n_features_to_select = 15)
rfe_fit = rfe.fit(x, y)
features = []
for feat in df_1.columns[rfe_fit.support_]:
    print(feat)
    features.append(feat)


# In[28]:


df_optm = df_new[features]


# In[29]:


df_optm.shape


# In[30]:


# Using Statsmodels for analysing the impact of attribute potential on the player rating:


# In[31]:


import statsmodels.formula.api as sm
model1 = sm.OLS(df_new['overall_rating'], df_new['potential'])
result1 = model1.fit()
print(result1.summary())


# In[32]:


# Using Statsmodels for analysing the impact of all attribute on the player rating


# In[34]:


X_new = df_new[features].values
model = sm.OLS(df_new['overall_rating'],df_new[features])
result = model.fit()
print(result.summary())


# In[ ]:


# Explanation of the OLS Regression Results :


# In[35]:


#Adjusted R-squared indicates that 99.9% of player ratings can be explained by our predictor variable.
#The regression coefficient (coef) represents the change in the dependent variable resulting from a one unit change in the 
#predictor variable, all other variables being held constant.    
#In our model, a one unit increase in potential increases the rating by 0.4525.
#The standard error measures the accuracy of potential's coefficient by estimating the variation of the coefficient if the 
#same test were run on a different sample of our population.
#Our standard error,0.001, is low and therefore appears accurate.
#The p-value means the probability of an 0.4525 increasing in player rating due to a one unit increase in potential is 0% , 
#assuming there is no relationship between the two variables.
#A low p-value indicates that the results are statistically significant, that is in general the p-value is less than 0.05.
#The confidence interval is a range within which our coefficient is likely to fall. 
#We can be 95% confident that potentials's coefficient will be within our confidence interval, [0.450,0.455].


# In[36]:


# REGRESSION PLOTS:


# In[39]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(20, 12))
fig = sm.graphics.plot_partregress_grid(result, fig=fig)


# In[40]:


import statsmodels.formula.api as smf
# Include only TV and Radio model:

lm = smf.ols(formula = 'overall_rating ~  player_fifa_api_id + player_api_id +potential +heading_accuracy +short_passing +ball_control +acceleration +reactions +strength +marking +gk_diving +gk_handling +gk_kicking+gk_positioning +gk_reflexes', data=df_new).fit()
print('Confidence of the statsmodel for the input data :', lm.rsquared)


# In[41]:


df_new.head()


# In[42]:


df_optm.head()


# In[43]:


# Data Exploration using visualization:


# In[44]:


df_corr = df_new.corr()


# In[45]:


import seaborn as sns
sns.set_style('whitegrid')
plt.figure(figsize=(20, 8))
sns.heatmap(df_corr,annot=True)


# In[48]:


# create correlation matrix with absolute values

df_corr = df_new.corr().abs()

#select upper triangle of matrix:

up_tri = df_corr.where(np.triu(np.ones(df_corr.shape[1]), k=1).astype(np.bool))

#find all the features which have a correlation > 0.75 with other features.

corr_features = [column for column in up_tri.columns if any(up_tri[column]>0.75)]

# Print Correlated features:

print(corr_features)


# In[49]:


# Drop Correlated features:

df_no_corr = df_new.drop(corr_features, axis=1)
df_no_corr.head()


# In[50]:


len(df_no_corr.columns)


# In[51]:


#This shows that the feature selection API - sklearn.feature_selection.RFE has resulted in the same feature selection for top 15 
#features selected.

#Quant Features against Rating


# In[53]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15, 60))
val = df_optm.shape[1]
for idx in range(val):
    feature = df_optm.columns[idx]
    ax = fig.add_subplot(13, 3, idx+1)
    Xtmp = df_optm[feature]
    ax.scatter(Xtmp, y)
    ax.set_xlabel(feature)

plt.tight_layout()
plt.show()


# In[54]:


# Split the input data into training and test data


# In[55]:


from sklearn.model_selection import train_test_split

# Spliting 66.66% for train data and 33.33% for test data.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
print("x_train shape : ", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape : ", y_train.shape)
print("y_test shape : ", y_test.shape)


# In[56]:


# Applying Linear Regression Model


# In[57]:


lm = LinearRegression()

# To train the model

lm.fit(x_train, y_train)


# In[58]:


# Perform Prediction using Linear Regression Model


# In[59]:


# Predict the prices based on the test data

y_pred = lm.predict(x_test)


# In[60]:


y_pred


# In[61]:


print("The variance score of the LinearRegression model is : ", lm.score(x_test, y_test))


# In[62]:


# Since the variance score is near about 1 looks to be a perfect prediction.


# In[63]:


import matplotlib.pyplot as plt
plt.figure(figsize = (6, 4))
plt.hist(y_pred)
plt.xlabel('Predicted Rating of the Player')
plt.ylabel('count')
plt.tight_layout()


# In[64]:


import seaborn as sns
sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.plot([0, 100], [0, 100], '--k')
plt.axis('tight')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.tight_layout()
plt.title("Actual vs Predicted Rating for LinearRegression Model")


# In[65]:


# Evaluate Linear Regression Accuracy using Root Mean Square Error


# In[66]:


from sklearn.metrics import mean_squared_error
print("Error Rate of the Regression Model : ", sqrt(mean_squared_error(y_pred, y_test)))


# In[67]:


# Applying Decision Tree Regressor Model to the input data


# In[68]:


regressor = DecisionTreeRegressor(max_depth=20)
regressor.fit(x_train, y_train)


# In[69]:


# Perform Prediction using Decision Tree Regressor


# In[70]:


y_pred = regressor.predict(x_test)


# In[71]:


y_pred


# In[73]:


print("The variance score of the DecisionTreeRegressor model is : ", regressor.score(x_test, y_test))


# In[74]:


# Here the variance score is nearly 1 which appears to be a perfect prediction


# In[75]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.hist(y_pred)
plt.xlabel('Predicted Rating of the Player')
plt.ylabel('count')
plt.tight_layout()


# In[76]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.plot([0, 100], [0, 100], '--k')
plt.axis('tight')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.tight_layout()
plt.title("Actual vs Predicted Rating for DecisionTreeRegressor Model")


# In[77]:


#The mean of the expected target value in test set:


# In[78]:


y_test.mean()


# In[79]:


# What is the mean of the predicted target value in test set ?


# In[80]:


y_pred.mean()


# In[81]:


# Evaluate Linear Regression Accuracy using Root Mean Square Error For DecisionTreeRegressor model:


# In[82]:


print("Error Rate of the DecisionTreeRegressor Model : ", sqrt(mean_squared_error(y_pred, y_test)))


# In[83]:


# Thus the DecisionTreeRegressor Model performs better than the LinearRegression Model as eveident from the error rate


# In[84]:


# Obtaining predictions by cross-validation for the Regression Models


# In[85]:


df_optm = df_new.copy()
df_optm['rating'] = y
df_optm.head()


# In[96]:


from sklearn.model_selection import cross_val_predict
X = df_optm.drop('rating', axis=1)
Y = df_optm['rating']
predicted = cross_val_predict(regressor, X, Y, cv=10)


# In[87]:


from sklearn.metrics import accuracy_score
print("Accuracy Score of the DecisionTreeRegressor Model is : ", accuracy_score(y.astype(int), predicted))


# In[88]:


# Calculate Error using K-Fold Cross validation


# In[97]:


from sklearn.cross_validation import KFold
kfold = KFold(len(df_optm), n_folds=10, shuffle=True, random_state=0)


# In[99]:


from sklearn.metrics import mean_absolute_error
lm = LinearRegression()
mean_abs_error = []
accuracy_score = []
for train,test in kfold:
    x = X.iloc[train]
    y = Y.iloc[train]
    lm.fit(x,y)
    Y_test = Y.iloc[test]
    Y_pred = lm.predict(X.iloc[test])
    mean_abs_error.append(mean_absolute_error(Y_test,Y_pred))


# In[100]:


print('10 Fold Cross Validation Error : {} accuracy score: {} for LinearRegression Model'.format(np.mean(mean_abs_error), 1 - np.mean(mean_abs_error)))


# In[101]:


from sklearn.metrics import mean_absolute_error

#DR = LinearRegression()

mean_abs_error = []
accuracy_score = []
for train,test in kfold:
    x = X.iloc[train]
    y = Y.iloc[train]
    regressor.fit(x, y)
    Y_test = Y.iloc[test]
    Y_pred = regressor.predict(X.iloc[test])
    mean_abs_error.append(mean_absolute_error(Y_test, Y_pred))


# In[102]:


print('10 Fold Cross Validation Error : {} accuracy score : {} for DecisionTreeRegressor Model'.format(np.mean(mean_abs_error), 1 - np.mean(mean_abs_error)))


# In[ ]:


# CONCLUSION :

# Have used the below models to predict the player ratings.

#Statsmodels.api.OLS
#LinearRegression
#DecisionTreeRegressor

# Sample mechanism used:

# Test Train Split
# 10 Fold Cross Validation

# Model Estimation mechanism used:

# Root Mean Squared Error.
# 10 Fold Cross Validation Error.


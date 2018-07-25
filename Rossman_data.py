
# coding: utf-8

# In[36]:


import warnings
warnings.filterwarnings("ignore")

# loading packages
# basic + dates 
import numpy as np
import pandas as pd
from pandas import datetime

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
get_ipython().magic('matplotlib inline')

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# prophet by Facebook
from fbprophet import Prophet

# machine learning: XGB
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor # wrapper


# In[37]:


import os 
os.chdir('C:/Uconn MSBA/studies/Kaggle/data and code/Time Series/Rossmann Store Sales')


# In[38]:


train  = pd.read_csv("train.csv",index_col = 'Date')
test_df      = pd.read_csv("test.csv")
store     = pd.read_csv("store.csv")


# In[39]:


train.index


# In[42]:


train.index=  pd.to_datetime(train.index)


# In[43]:


train.head()


# In[44]:


# data extraction
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear

# adding new variable
train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].describe()


# # Visualisations

# In[45]:


print(train.dtypes)


# In[46]:


# Checking if all the stores are present in trainign data 

train.Store.unique()


# In[47]:


# Converting character 0 to 0 

rossmann_df_update =train

rossmann_df_update["StateHoliday"].loc[rossmann_df_update["StateHoliday"] == 0] = "0"


# In[48]:


# plotting the data after changing the code 
sns.countplot(x="StateHoliday",data =rossmann_df_update)  


# In[49]:


#sales with respect to day of the week

sns.factorplot(x="DayOfWeek" , y = "Sales" ,hue="Open",data=rossmann_df_update )


# In[50]:


# sales with respect to the stateholiday

sns.barplot(x='StateHoliday', y='Sales', data=rossmann_df_update)


# In[69]:


# Open
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.countplot(x='Open',hue='DayOfWeek', data=train, ax=axis1)


# In[14]:


#State Holiday 
rossmann_df_update["StateHoliday"] = rossmann_df_update["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test_df["StateHoliday"]     = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})


# In[15]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StateHoliday', y='Sales', data=rossmann_df_update, ax=axis1)
sns.barplot(x='StateHoliday', y='Customers', data=rossmann_df_update, ax=axis2)


# # average_sales

# In[18]:


# group by date and get average sales, and precent change
average_sales    = rossmann_df_update.groupby('Date')["Sales"].mean()
pct_change_sales = rossmann_df_update.groupby('Date')["Sales"].sum().pct_change()





# In[19]:


fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))

# plot average sales over time(year-month)
ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist())

# plot precent change for sales over time(year-month)
ax2 = pct_change_sales.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales Percent Change")


# In[20]:


# Plot average sales & customers for every year
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Year', y='Sales', data=rossmann_df_update, ax=axis1)
sns.barplot(x='Year', y='Customers', data=rossmann_df_update, ax=axis2)


# In[21]:


# Plot max, min values, & 2nd, 3rd quartile
sns.boxplot([rossmann_df["Customers"]], whis=np.inf)


# # Customers

# In[22]:


fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))

# Plot max, min values, & 2nd, 3rd quartile
sns.boxplot([rossmann_df_update["Customers"]], whis=np.inf, ax=axis1)


# In[23]:


# Customers

fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))

# Plot max, min values, & 2nd, 3rd quartile
sns.boxplot([rossmann_df_update["Customers"]], whis=np.inf, ax=axis1)

# group by date and get average customers, and precent change
average_customers      = rossmann_df_update.groupby('Date')["Customers"].mean()
# pct_change_customers = rossmann_df.groupby('Date')["Customers"].sum().pct_change()

# Plot average customers over the time
# it should be correlated with the average sales over time
ax = average_customers.plot(legend=True,marker='o', ax=axis2)
ax.set_xticks(range(len(average_customers)))
xlabels = ax.set_xticklabels(average_customers.index.tolist(), rotation=90)


# # DayOfWeek
# 

# In[24]:


# In both cases where the store is closed and opened

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='DayOfWeek', y='Sales', data=rossmann_df_update, order=[1,2,3,4,5,6,7], ax=axis1)
sns.barplot(x='DayOfWeek', y='Customers', data=rossmann_df_update, order=[1,2,3,4,5,6,7], ax=axis2)


# In[25]:


# Promo

# Plot average sales & customers with/without promo
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Promo', y='Sales', data=rossmann_df_update, ax=axis1)
sns.barplot(x='Promo', y='Customers', data=rossmann_df_update, ax=axis2)


# In[26]:


sns.pointplot(x="Customers",y='Sales',data=rossmann_df_update)


# In[28]:


rossmann_df.head()


# # Forecasting 

# In[52]:


df = train


# In[60]:


df['Date'] = df.index


# In[61]:


# remove closed stores and those with no sales
df = df[(df["Open"] != 0) & (df['Sales'] != 0)]

# sales for the store number 1 (StoreType C)
sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

# to datetime64
sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales.dtypes


# In[62]:



# from the prophet documentation every variables should have specific names# from t 
sales = sales.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})
sales.head()


# In[63]:


df['index1'] = df.index


# In[64]:


# create holidays dataframe
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b') & (df.StateHoliday == 'c')].loc[:, 'Date'].values
school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                      'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))      
holidays.head()


# In[65]:


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width = 0.95, 
                   holidays = holidays)
my_model.fit(sales)

# dataframe that extends into future 6 weeks 
future_dates = my_model.make_future_dataframe(periods = 6*7)

print("First week to forecast.")
future_dates.tail(7)


# In[66]:


# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)


# In[67]:


fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})


# In[68]:


# visualizing predicions
my_model.plot(forecast);


# In[ ]:



my_modelmy_model.plot_components(forecast);


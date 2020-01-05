
# coding: utf-8

#  # AD Click Prediction

# In[1]:


## importing libraries ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import featuretools as ft
warnings.filterwarnings('ignore')
sns.set(style = "whitegrid")


# In[2]:


data = pd.read_csv('train.csv', index_col= ['DateTime'], parse_dates= True)


# In[3]:


data.info()


# #### Visualizing for missing values.

# In[4]:


sns.heatmap(data.isnull())


# ### Data Imputation

# In[5]:


# data imputation
data = data.drop('product_category_2', axis = 1) # dropping the column 
# for rest of the columns with missing values, imputing using forward fill.
data['city_development_index'] = data['city_development_index'].fillna(method = 'ffill') 
data['gender'] = data['gender'].fillna(method = 'ffill')
data['user_group_id'] = data['user_group_id'].fillna(method = 'ffill')
data['age_level'] = data['age_level'].fillna(method = 'ffill')
data['user_depth'] = data['user_depth'].fillna(method = 'ffill')


# In[6]:


data.head()


# In[7]:


data.info()


# ### Data Visualization

# #### Visualizing the trends in the data by setting granularity to per hour on daily basis.

# In[8]:


day = data.groupby('DateTime')['is_click'].sum()
day = day.resample('H').sum()
plt.figure(figsize=(20,5))
day.plot(kind='bar',grid = None)


# #### Portion of the above plot for 2 day of user data.

# In[9]:


part_day = day.loc[slice('2017-07-02','2017-07-03')]
plt.figure(figsize=(20,5))
part_day.plot(kind='bar',grid = None)


# #### Visualizing the user behavior on weekday basis. It seems that most of the clicks are for MONDAY & SUNDAY.

# In[10]:


data1 = data.reset_index()
data1['weekday'] = data1['DateTime'].dt.day_name()
byday  = pd.DataFrame(data1.groupby('weekday')['is_click'].sum())
byday = byday.reset_index()
plt.figure(figsize=(20,5))
sns.barplot(data = byday , x= 'weekday', y = 'is_click')


# #### Visualizing data for different products for male and female user groups.

# In[11]:


user = data.groupby(['gender','product'])['is_click'].sum()
user = pd.DataFrame(user.reset_index())
plt.figure(figsize=(20,5))
sns.barplot(data = user, x= 'product', y = 'is_click', hue = 'gender',palette='Set1')


# #### Barplot showing max clicks for all the product from a single campaign.

# In[12]:


n_data = data.reset_index()
campaign= pd.DataFrame(n_data.groupby(['campaign_id','product'])['is_click'].sum())
campaign= campaign.reset_index()
campaign= campaign.groupby(['product'])[['campaign_id','is_click']].max()
campaign= campaign.sort_values('is_click',ascending = False).reset_index()
campaign.columns = ['product', 'campaign_id', 'max click in any campaign']
plt.figure(figsize=(15,5))
sns.barplot(y= 'product', x= 'max click in any campaign', palette = 'Set1', data = campaign, orient='h')


# #### Table highlighting the most successful product and no. of clicks for each of them for each campaign.

# In[13]:


n_data = data.reset_index()
campaign= pd.DataFrame(n_data.groupby(['campaign_id','product'])['is_click'].sum())
campaign= campaign.reset_index()
campaign= campaign.groupby('campaign_id')[['product','is_click']].max()
campaign.sort_values('is_click',ascending = False)


# #### This visualization highlights that all the user groups from 0-6 are Male and from 7-12 are Females.

# In[14]:


plt.figure(figsize=(20,5))
sns.countplot(x= 'user_group_id', hue= 'gender', palette = 'Set1', data = data)


# #### successs % on the basis of the user id group. Most successful user group is 12.

# In[15]:


plt.figure(figsize=(15,5))
user_group = data.groupby('user_group_id')['is_click'].agg(['count','sum'])
user_group['%success']= (user_group['sum']*100)/user_group['count']
user_group = user_group.reset_index()
sns.barplot(y= 'user_group_id', x= '%success', palette = 'Set1', data = user_group, orient='h')


# #### Visualizing count of clicks and non clicks for each of the product.

# In[16]:


plt.figure(figsize=(15,3))
sns.countplot(x="product", hue= "is_click", palette = 'Set1',data =data )


# #### Performance of all the products compared category wise.

# In[17]:


plt.figure(figsize=(15,5))
sns.countplot(x="product", hue= "product_category_1", palette = 'Set1', data =data)


# In[18]:


data1 = data[['user_depth', 'is_click']]
data1 = data.groupby(['user_depth','is_click']).size().unstack()
data1['success %'] = round(data1[1]*100/(data1[1]+data1[0]),2)
data1


# In[19]:


# data to be used for the purpose of modelling
finaldata = data.reset_index()  


# ### Feature Engineering & Data Preprocessing

# In[20]:


finaldata['weekday']=  finaldata['DateTime'].dt.day_name()
finaldata['hour'] = finaldata['DateTime'].dt.hour
finaldata['minutes'] = finaldata['DateTime'].dt.minute
finaldata = finaldata.drop('DateTime', axis = 1)
finaldata.head()


# In[21]:


print(finaldata['is_click'].value_counts())
print(round(30057*100/(414991),2))   # Number of clicks is only 7.24%. So, our dataset is imbalanced.


# In[22]:


# user id with maximum advertisement views of the all the campaigns combined.
#print((data['user_id'].value_counts()))

exp1 = data[data['user_id'].isin([658554])]
print(exp1['is_click'].value_counts())  
# Ever for this user, the success rate is only 1.17%. So, more advertisement views does not ensure more clicks.


# In[23]:


finaldata = pd.get_dummies(finaldata, columns =['product','campaign_id','product_category_1','user_group_id','gender','age_level','user_depth','var_1','weekday','hour','city_development_index'], drop_first= True)
finaldata.info()


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X = finaldata.drop('is_click', axis =1)
y = finaldata['is_click']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# ##### Balacing the data set

# In[26]:


from imblearn.over_sampling import SMOTE


# In[27]:


sm = SMOTE(random_state = 101)
X_train, y_train  = sm.fit_sample(X_train,y_train)


# ### Data Modelling

# #### Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


logreg  = LogisticRegression()
logreg.fit(X_train,y_train)


# In[30]:


prediction  = logreg.predict(X_test)


# In[31]:


from sklearn import metrics
from sklearn.metrics import classification_report


# In[32]:


matrix1 = classification_report(y_test, prediction)
print(matrix1)


# In[33]:


matrix2 = metrics.roc_auc_score(y_test,prediction)
print(matrix2)


# #### Decision Tree

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


dctree = DecisionTreeClassifier()


# In[36]:


dctree.fit(X_train, y_train)


# In[37]:


# List of most important features in Descending order.
pd.Series(dctree.feature_importances_, index= X.columns).sort_values(ascending=False)


# In[38]:


prediction = dctree.predict(X_test)


# In[39]:


matrix1 = classification_report(y_test, prediction)
print(matrix1)


# In[40]:


matrix2 = metrics.roc_auc_score(y_test, prediction)
print(matrix2)


# #### Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rf = RandomForestClassifier()


# In[43]:


rf.fit(X_train, y_train)


# In[44]:


prediction = rf.predict(X_test)


# In[45]:


matrix1 = classification_report(y_test, prediction)
print(matrix1)


# In[47]:


matrix2 = metrics.roc_auc_score(y_test, prediction)
print(matrix2)


# #### Multi Layer Perceptron

# In[48]:


from sklearn.neural_network import MLPClassifier


# In[49]:


mlp = MLPClassifier(hidden_layer_sizes=(100,3), alpha = 0, max_iter=200)


# In[50]:


mlp.fit(X_train, y_train)


# In[51]:


prediction = mlp.predict(X_test)


# In[52]:


matrix2 = metrics.roc_auc_score(y_test, prediction)
print(matrix2)


# #### Boosting (XGBoost)

# In[53]:


from xgboost import XGBClassifier


# In[54]:


xgb = XGBClassifier()


# In[55]:


xgb.fit(X_train, y_train)


# In[56]:


prediction = xgb.predict(X_test.values)


# In[57]:


matrix1 = classification_report(y_test, prediction)
print(matrix1)


# In[58]:


matrix2 = metrics.roc_auc_score(y_test, prediction)
print(matrix2)


# ### Through this project, I learnt the concept of Feature Engineering. I was able to identify the trend in the user behavior by manipulating the datetime variable.

#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[47]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans


# In[48]:


df = pd.read_csv('Mall_Customers.csv')


# # Data Anlaysis

# In[49]:


df.head()


# In[50]:


df.tail()


# In[51]:


df.shape

# If we have low amount of datapoints,
# how can we enlarge the present datapoints to train our model


# In[52]:


df.isnull().sum()


# In[53]:


df.info()


# * Conclusions
# 
# 1. No Null Value
# 2. Gender has the dtype of object, which needs to be taken care of

# In[54]:


df.describe()


# * Conclusions
# 
# 1. Avg. age of Customer is 38
# 2. Customer that visits the mall is having the age b/w 18 to 70
# 3. Having the Avg. annual income as $60k

# In[55]:


plt.figure(figsize=(20,15))
sns.countplot(data=df, x='Age')


# In[56]:


df.head()


# In[57]:


plt.figure(figsize=(8,8))
df.Gender.value_counts().plot(kind='pie', autopct='%.2f%%', shadow=True,
                              explode=(0,0.04))
plt.legend()


# * Conclusion
# 
# 1. No. of females > No. of males
# 2. Females are 12% more than Male
# 3. Female visits the mall most of the time.

# In[58]:


df.head(2)


# In[59]:


# Distribution of Income
plt.figure(figsize=(20,7))
sns.countplot(df['Annual Income (k$)'], data=df)
plt.title('Distribution of Anuual Income')


# * Conclusions
# 
# 1. The customers that are having the salary of $54(k) and $78(K), visits the mall most of time

# In[60]:


# Show the Annual Income Distribution w.r.t. to the Gender

plt.figure(figsize=(6,6))
sns.histplot(df,x='Annual Income (k$)',hue='Gender', multiple='stack')
plt.title('Annual Income Distribution w.r.t. to the Gender')
plt.grid()


# * Conclusion

# # Seaborn lmplot

# In[61]:


df.head()


# In[62]:


sns.lmplot(data=df,x='Spending Score (1-100)', y='Age', hue='Gender')


# In[63]:


df.head()


# In[64]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder


# In[65]:


enc = LabelEncoder()


# In[66]:


df.Gender = enc.fit_transform(df.Gender)


# In[67]:


df.head()


# In[68]:


df.drop('CustomerID', axis=1, inplace=True)


# In[69]:


ssd = []
for i in range(1,11):
  Kmodel = KMeans(n_clusters=i, n_init=15,max_iter=500)
  Kmodel.fit(df)
  ssd.append(Kmodel.inertia_)


# In[70]:


ssd


# In[71]:


plt.plot(range(1,11), ssd, marker='o')
plt.grid()
plt.title('Elbow plot')


# In[72]:


# k=6


# In[73]:


Kmodel = KMeans(n_clusters=6)


# In[74]:


Kmodel.fit(df)
# Kmodel.fit_predict(df)
# Kmodel.predict(df)


# In[75]:


prediction = Kmodel.predict(df)


# In[76]:


prediction


# In[77]:


len(prediction)


# In[78]:


Kmodel.cluster_centers_


# In[79]:


len(Kmodel.cluster_centers_)


# In[80]:


df.head()


# In[81]:


df['Cluster'] = prediction


# In[82]:


df.head()


# In[83]:


color = np.array(['darkgray', 'lightsalmon', 'powderblue', 'red', 'yellow', 'black'])
sns.scatterplot(x = df['Annual Income (k$)'],
                y = df['Spending Score (1-100)'],
                c=color[Kmodel.labels_],
                s=50)


# In[85]:


from scipy.cluster.hierarchy import linkage, dendrogram


# In[86]:


var = linkage(df, method='ward')


# In[87]:


plt.figure(figsize=(20,15))
dendrogram(var, leaf_rotation=90)


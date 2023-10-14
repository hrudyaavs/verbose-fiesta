#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[74]:


df = pd.read_csv("D:/Anacoda/iris.csv")
df.head(10)


# Columns names 

# In[64]:


df.columns


# Mean of each Columns 

# In[86]:


C1=df["SL"].mean()
C2=df["SW"].mean()
C3=df["PL"].mean()
C4=df["PW"].mean()
print("SL : ",C1,"\nSW : ",C2,"\nPL : ",C3,"\nPW : ",C4)


# In[85]:


C5=df['Classification'].value_counts()
print("\nClassification Column Count of each Species\n",C5)


# Null values present or not 

# In[54]:


df.isnull()


# # Visualizations 

# Bar Chart

# In[66]:


sns.countplot(x='Classification',data=df)
plt.show()


# Scatter Plot

# In[67]:


sns.scatterplot(x='SL', y='SW',
                hue='Classification', data=df, )

plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# Histograms

# In[70]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))
 
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['SL'], bins=7)
 
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['SW'], bins=5);
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['PL'], bins=6);
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['PW'], bins=6);


# Heatmap

# In[59]:


sns.heatmap(df.corr())


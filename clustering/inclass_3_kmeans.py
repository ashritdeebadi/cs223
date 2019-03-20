
# coding: utf-8

# In[1]:


import seaborn as sns


# In[3]:


import pandas as pd


# In[10]:


df=pd.DataFrame.from_csv(path='C:/Users/user/Desktop/223/inclass_3.csv')


# In[11]:


df


# In[12]:


sns.pairplot(df)


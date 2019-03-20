
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


values=[(-3,1),(3.5,1),(4,1),(5,1),(6,1),(7.7,1),(9,1),(10,1),(14,1)]
X=np.array(values)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[9]:


kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[10]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')


# In[16]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers=kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[40]:





# In[60]:





# In[52]:





# In[58]:





# In[34]:





# In[24]:





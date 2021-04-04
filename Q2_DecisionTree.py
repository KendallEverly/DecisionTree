#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets


# In[2]:


import pandas as pd 
data=pd.read_csv("C:/Users/kenda/Downloads/sample.csv")
data.head(5)


# In[6]:


x=data[["blue", "four_g", "n_cores"]]
y=data[["price_range"]]

print(x)
print(y)


# In[7]:


dtree=DecisionTreeClassifier(random_state=0)
dtree=dtree.fit(x,y)


# In[9]:


NewObservation=[[1,0,5]]
dtree.predict(NewObservation)
print(NewObservation)


# In[10]:


import pydotplus


# In[11]:


from IPython.display import Image
from sklearn import tree


# In[26]:


dot_data=tree.export_graphviz(dtree, out_file=None, feature_names=None)
graph=pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())


# In[ ]:





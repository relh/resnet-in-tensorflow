
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

#data=pd.read_csv('out.csv')


# In[3]:

probs=np.zeros((9300,14))
labels=np.zeros((9300,14))
with open('out.csv', 'r') as f:
    for i in range(9300):
        temp=f.readline().split()
        for j in range(1,7):
            probs[i,j-1]=float(temp[j])
        temp=f.readline().split()
        for j in range(0,6):
            probs[i,j+6]=float(temp[j])
        temp=f.readline().split()
        if len(temp)==16:
            probs[i,12]=float(temp[0])
            probs[i,13]=float(temp[1][:-3])
            for j in range(13):
                labels[i,j]=float(temp[j+2])
            labels[i,13]=float(temp[-1][:-1])
        else:
            probs[i,12]=float(temp[0])
            probs[i,13]=float(temp[1])
            for j in range(13):
                labels[i,j]=float(temp[j+3])
            labels[i,13]=float(temp[-1][:-1])
        #for j=range(1,7):
        #    probs(i,j-1)=float(temp)


# In[4]:

from sklearn.metrics import roc_auc_score


# In[5]:

for i in range(14):
    print(roc_auc_score(labels[:,i], probs[:,i]))


# In[43]:




# In[ ]:




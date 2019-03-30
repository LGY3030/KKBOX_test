
# coding: utf-8

# In[19]:


from keras.utils import np_utils
import numpy as np
import pandas as pd


# In[11]:


train=pd.read_csv("data_processed/train.csv")
test=pd.read_csv("data_processed/test.csv")
target=pd.read_csv("data_processed/target.csv")


# In[ ]:


train=train.drop(["Unnamed: 0"], axis=1)
test=test.drop(["Unnamed: 0"], axis=1)
target=target.drop(["Unnamed: 0"], axis=1)


# In[12]:


train=train.convert_objects(convert_numeric=True)
test=test.convert_objects(convert_numeric=True)
target=target.convert_objects(convert_numeric=True)


# In[15]:


train=(train-train.mean())/train.std()
test=(test-test.mean())/test.std()


# In[23]:


train=np.array(train)
test=np.array(test)
target=np.array(target)


# In[24]:


print(train.shape)
print(test.shape)
print(target.shape)


# In[21]:


target=np_utils.to_categorical(target)
print(target)


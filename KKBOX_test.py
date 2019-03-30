#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd


# In[42]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv", index_col=0)
songs=pd.read_csv("songs.csv")
members=pd.read_csv("members.csv")
song_extra_info=pd.read_csv("song_extra_info.csv")
train=train.merge(songs, on='song_id', how='left')
train=train.merge(members, on='msno', how='left')
train["gender"]=train["gender"].map({'male':1,'female':2})
train["gender"]=train["gender"].fillna(0)
test=test.merge(songs, on='song_id', how='left')
test=test.merge(members, on='msno', how='left')
test["gender"]=test["gender"].map({'male':1,'female':2})
test["gender"]=test["gender"].fillna(0)


# In[43]:


train["registration_init_time"] = pd.to_datetime(train["registration_init_time"],format='%Y%m%d')
train["registration_init_time_year"] = train["registration_init_time"].dt.year
train["registration_init_time_month"] = train["registration_init_time"].dt.month
train["registration_init_time_day"] = train["registration_init_time"].dt.day
train["registration_init_time_dayofweek"] = train["registration_init_time"].dt.dayofweek
train=train.drop(["registration_init_time"], axis=1)
train["expiration_date"] = pd.to_datetime(train["expiration_date"],format='%Y%m%d')
train["expiration_date_year"] = train["expiration_date"].dt.year
train["expiration_date_month"] = train["expiration_date"].dt.month
train["expiration_date_day"] = train["expiration_date"].dt.day
train["expiration_date_dayofweek"] = train["expiration_date"].dt.dayofweek
train=train.drop(["expiration_date"], axis=1)
test["registration_init_time"] = pd.to_datetime(test["registration_init_time"],format='%Y%m%d')
test["registration_init_time_year"] = test["registration_init_time"].dt.year
test["registration_init_time_month"] = test["registration_init_time"].dt.month
test["registration_init_time_day"] = test["registration_init_time"].dt.day
test["registration_init_time_dayofweek"] = test["registration_init_time"].dt.dayofweek
test=test.drop(["registration_init_time"], axis=1)
test["expiration_date"] = pd.to_datetime(test["expiration_date"],format='%Y%m%d')
test["expiration_date_year"] = test["expiration_date"].dt.year
test["expiration_date_month"] = test["expiration_date"].dt.month
test["expiration_date_day"] = test["expiration_date"].dt.day
test["expiration_date_dayofweek"] = test["expiration_date"].dt.dayofweek
test=test.drop(["expiration_date"], axis=1)


# In[44]:


print(train)


# In[45]:


temp = train['source_system_tab'].value_counts().rename_axis('source_system_tab').reset_index(name='Number')
print(temp)
'''
'''


# In[46]:


print(temp["source_system_tab"])


# In[52]:


temp2=train[:]
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["source_system_tab"][i]]=i
temp2["source_system_tab"]=temp2["source_system_tab"].map(mapmap)
print(temp2)


# In[48]:


print(temp["source_system_tab"][0])


# In[ ]:





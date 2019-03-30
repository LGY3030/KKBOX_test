
# coding: utf-8

# In[55]:


import pandas as pd


# In[56]:


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


# In[57]:


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


# In[58]:


temp=pd.concat([train['source_system_tab'],test['source_system_tab']],axis=0)
temp = temp.value_counts().rename_axis('source_system_tab').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["source_system_tab"][i]]=i+1
train["source_system_tab"]=train["source_system_tab"].map(mapmap)
test["source_system_tab"]=test["source_system_tab"].map(mapmap)
train["source_system_tab"]=train["source_system_tab"].fillna(0)
test["source_system_tab"]=test["source_system_tab"].fillna(0)


# In[59]:


temp=pd.concat([train['source_screen_name'],test['source_screen_name']],axis=0)
temp = temp.value_counts().rename_axis('source_screen_name').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["source_screen_name"][i]]=i+1
train["source_screen_name"]=train["source_screen_name"].map(mapmap)
test["source_screen_name"]=test["source_screen_name"].map(mapmap)
train["source_screen_name"]=train["source_screen_name"].fillna(0)
test["source_screen_name"]=test["source_screen_name"].fillna(0)


# In[60]:


temp=pd.concat([train['source_type'],test['source_type']],axis=0)
temp = temp.value_counts().rename_axis('source_type').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["source_type"][i]]=i+1
train["source_type"]=train["source_type"].map(mapmap)
test["source_type"]=test["source_type"].map(mapmap)
train["source_type"]=train["source_type"].fillna(0)
test["source_type"]=test["source_type"].fillna(0)


# In[61]:


temp=pd.concat([train['song_id'],test['song_id']],axis=0)
temp = temp.value_counts().rename_axis('song_id').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["song_id"][i]]=i+1
train["song_id"]=train["song_id"].map(mapmap)
test["song_id"]=test["song_id"].map(mapmap)
train["song_id"]=train["song_id"].fillna(0)
test["song_id"]=test["song_id"].fillna(0)


# In[62]:


temp=pd.concat([train['msno'],test['msno']],axis=0)
temp = temp.value_counts().rename_axis('msno').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["msno"][i]]=i+1
train["msno"]=train["msno"].map(mapmap)
test["msno"]=test["msno"].map(mapmap)
train["msno"]=train["msno"].fillna(0)
test["msno"]=test["msno"].fillna(0)


# In[63]:


temp=pd.concat([train['artist_name'],test['artist_name']],axis=0)
temp = temp.value_counts().rename_axis('artist_name').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["artist_name"][i]]=i+1
train["artist_name"]=train["artist_name"].map(mapmap)
test["artist_name"]=test["artist_name"].map(mapmap)
train["artist_name"]=train["artist_name"].fillna(0)
test["artist_name"]=test["artist_name"].fillna(0)


# In[64]:


temp=pd.concat([train['composer'],test['composer']],axis=0)
temp = temp.value_counts().rename_axis('composer').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["composer"][i]]=i+1
train["composer"]=train["composer"].map(mapmap)
test["composer"]=test["composer"].map(mapmap)
train["composer"]=train["composer"].fillna(0)
test["composer"]=test["composer"].fillna(0)


# In[65]:


temp=pd.concat([train['genre_ids'],test['genre_ids']],axis=0)
temp = temp.value_counts().rename_axis('genre_ids').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["genre_ids"][i]]=i+1
train["genre_ids"]=train["genre_ids"].map(mapmap)
test["genre_ids"]=test["genre_ids"].map(mapmap)
train["genre_ids"]=train["genre_ids"].fillna(0)
test["genre_ids"]=test["genre_ids"].fillna(0)


# In[66]:


temp=pd.concat([train['lyricist'],test['lyricist']],axis=0)
temp = temp.value_counts().rename_axis('lyricist').reset_index(name='Number')
mapmap={}
for i in range(temp.shape[0]):
    mapmap[temp["lyricist"][i]]=i+1
train["lyricist"]=train["lyricist"].map(mapmap)
test["lyricist"]=test["lyricist"].map(mapmap)
train["lyricist"]=train["lyricist"].fillna(0)
test["lyricist"]=test["lyricist"].fillna(0)


# In[67]:


column_list=list(train.columns.values)
column_list.remove('target')
target=train.drop(column_list, axis=1)
train=train.drop(["target"], axis=1)


# In[68]:


train.to_csv("data_processed/train.csv", encoding='utf_8_sig')
test.to_csv("data_processed/test.csv", encoding='utf_8_sig')
target.to_csv("data_processed/target.csv", encoding='utf_8_sig')


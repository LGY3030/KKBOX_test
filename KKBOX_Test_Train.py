
# coding: utf-8

# In[17]:


from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# In[18]:


train=pd.read_csv("data_processed/train.csv")
test=pd.read_csv("data_processed/test.csv")
target=pd.read_csv("data_processed/target.csv")
train=train.fillna(0)
target=target.fillna(0)
test=test.fillna(0)


# In[19]:


train=train.drop(["Unnamed: 0"], axis=1)
test=test.drop(["Unnamed: 0"], axis=1)
target=target.drop(["Unnamed: 0"], axis=1)


# In[20]:


train=train.convert_objects(convert_numeric=True)
test=test.convert_objects(convert_numeric=True)
target=target.convert_objects(convert_numeric=True)


# In[21]:


train=(train-train.mean())/train.std()
test=(test-test.mean())/test.std()


# In[22]:


train=np.array(train)
test=np.array(test)
target=np.array(target)
target=target.reshape(-1)


# In[23]:


model=Sequential()
model.add(Dense(units=100,input_dim=23,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=50,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0005),metrics=['accuracy'])
train_history=model.fit(x=train,y=target,validation_split=0.1,epochs=30,batch_size=30,verbose=1)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig('result.jpg')
    plt.clf()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
result=pd.DataFrame({'acc':train_history.history['acc'],'val_acc':train_history.history['val_acc'],'loss':train_history.history['loss'],'val_loss':train_history.history['val_loss']})
result.to_csv("target.csv", encoding='utf_8_sig')
model.save('model.h5')


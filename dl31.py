#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fashion_train_df= pd.read_csv('/content/drive/MyDrive/archive/fashion-mnist_train.csv')


# In[ ]:


fashion_test_df = pd.read_csv('/content/drive/MyDrive/archive/fashion-mnist_test.csv')


# In[ ]:


fashion_train_df.head()


# In[ ]:


fashion_train_df.tail()


# In[ ]:


fashion_train_df.shape


# In[ ]:


fashion_test_df.shape


# In[ ]:


training = np.array(fashion_train_df,dtype='float32')
testing = np.array(fashion_test_df,dtype='float32')


# In[ ]:


training.shape


# In[ ]:


import random


# In[ ]:


i = random.randint(0,60001) 
plt.imshow(training[i,1:].reshape(28,28)) 
label = training[i,1] 
label


# **i** = random.randint(0,60001)
# plt.imshow(training[i,1:].reshape(28,28))
# label = training[i,1] 
# label

# In[ ]:


W_grid = 7
L_grid = 7

fig,axes = plt.subplots(L_grid,W_grid,figsize =(17,17))

axes = axes.ravel() 
n_training = len(training) 



for i in np.arange(0,W_grid*L_grid):
        index = np.random.randint(0,n_training)
        axes[i].imshow(training[index,1:].reshape((28,28)))
        axes[i].set_title(training[index,0],fontsize = 8)
        axes[i].axis('off')
        
plt.subplots_adjust(hspace=0.4)  


# In[ ]:


X_train = training[:,1:]/255
y_train = training[:,0]
X_test = testing[:,1:]/255
y_test = testing[:,0]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,test_size = 0.2,random_state = 12345) 


# In[ ]:


X_train = X_train.reshape(X_train.shape[0],*(28,28,1))
X_test = X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0],*(28,28,1))


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_validate.shape


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:


cnn_model = Sequential()
cnn_model.add(Conv2D(32,3,3,input_shape = (28,28,1),activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size= (2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(32,activation = 'relu'))
cnn_model.add(Dense(10,activation = 'sigmoid'))
cnn_model.compile(loss ='sparse_categorical_crossentropy',optimizer = Adam(learning_rate=0.001),metrics= ['accuracy'])


# In[ ]:


epochs = 200


# In[ ]:



cnn_model.fit(X_train,y_train,batch_size =512,epochs = epochs,verbose = 1,validation_data = (X_validate,y_validate) )


# In[ ]:


evaluation = cnn_model.evaluate(X_test,y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))


# In[ ]:


predicted_classes = np.argmax(cnn_model.predict(X_test),axis=-1)


# In[ ]:


predicted_classes


# In[ ]:


L = 5
W = 5

fig,axes = plt.subplots(L,W,figsize = (12,12))
axes = axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title('Prediction Class:{1} \n true class: {1}'.format(predicted_classes[i],y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 0.5)  


# In[ ]:


from sklearn.metrics import classification_report

classes = 10
targets = ["Class {}".format(i) for i in range(classes)]
print(classification_report(y_test, predicted_classes, target_names = targets))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.fashion_mnist.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[4]:


y_train.shape


# In[5]:


y_train[:5]


# In[6]:


classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# In[7]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[8]:


plot_sample(X_train, y_train, 0)


# In[9]:


plot_sample(X_train, y_train, 1)


# In[10]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[11]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[12]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[13]:


cnn.fit(X_train, y_train, epochs=10)


# In[14]:


cnn.evaluate(X_test,y_test)


# In[15]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[16]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[17]:


y_test[:5]


# In[18]:


plot_sample(X_test, y_test,3)


# In[19]:


classes[y_classes[3]]


# In[23]:


L = 5
W = 5

fig,axes = plt.subplots(L,W,figsize = (12,12))
axes = axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction Class:{1} \n true class: {1}'.format(y_classes[i],y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 0.5) 


# In[20]:


from sklearn.metrics import classification_report

classes = 10
targets = ["Class {}".format(i) for i in range(classes)]
print(classification_report(y_test, y_classes, target_names = targets))


# In[ ]:





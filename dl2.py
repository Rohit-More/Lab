#!/usr/bin/env python
# coding: utf-8

# In[59]:


from keras.datasets import imdb
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
from sklearn import metrics


# In[60]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import models


# In[61]:


(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


# In[62]:


X[0]


# In[63]:


y


# In[64]:


word_index = imdb.get_word_index()
word_index


# In[65]:


##training data shape review
print("Training data: ")
print(X.shape)
print(y.shape)
print("Classes: ")
print(np.unique(y))


# In[66]:


print("Number of words: ")
print(len(np.unique(np.hstack(X))))


# In[67]:


print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
plt.boxplot(result)
plt.show()


# In[68]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=5000)


# In[69]:


def vectorize_sequences(sequences, dimension=5000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# In[70]:


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)


# In[71]:


x_train.shape


# In[72]:


x_train


# In[73]:


# Our vectorized labels one-hot encoder
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[74]:


y_train


# In[75]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(5000,)))
model.add(layers.Dense(32, activation='relu',))
model.add(layers.Dense(1, activation='sigmoid'))


# In[76]:


#Set validation set aside

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[77]:


x_val.shape


# In[78]:


y_val.shape


# In[79]:


(partial_x_train.shape), (partial_y_train.shape)


# In[80]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[81]:


start_time_m1 = time.time()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
total_time_m1 = time.time() - start_time_m1


# In[82]:


print("The Dense Convolutional Neural Network 1 layer took %.4f seconds to train." % (total_time_m1))


# In[83]:


history_dict = history.history
history_dict.keys()


# In[84]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[85]:


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[86]:


model.summary()


# In[87]:


evaluation = model.evaluate(x_test,y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))


# In[88]:


from sklearn.metrics import confusion_matrix, accuracy_score, auc
#predictions
pred = model.predict(x_test)
classes_x=np.argmax(pred,axis=1)

#accuracy
accuracy_score(y_test,classes_x)


# In[89]:


#Confusion Matrix
conf_mat = confusion_matrix(y_test, classes_x)
print(conf_mat)

conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[90]:


r=metrics.classification_report(y_test, classes_x)
print(r)


# In[90]:





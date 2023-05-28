#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[2]:


# Read the dataset from a CSV file
df = pd.read_csv('Boston-house-price-data.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


# Finding out the correlation between the features
corr = df.corr()
corr.shape


# In[9]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)


# In[10]:


# Separate features and target variable
X = df.drop('MEDV', axis=1)
y = df['MEDV']


# In[11]:


X.shape


# In[12]:


y.shape


# In[13]:


# Visualize the target variable
sns.histplot(y, kde=True)
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable (MEDV)')
plt.show()


# In[14]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# Data cleaning
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


# Build the linear regression model using a deep neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[17]:


# Compile the model and specify the loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')


# In[18]:


# Train the model on the training data
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)


# In[19]:


# Evaluate the model on the test data
loss = model.evaluate(X_test_scaled, y_test, verbose=1)
print('Mean Squared Error on test set:', loss)


# In[20]:


# Predicting Test data with the model
y_test_pred = model.predict(X_test_scaled)


# In[21]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, y_test_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[22]:


# Model Evaluation
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[ ]:


# Outlier detection and removal
# Calculate the IQR (interquartile range) for each feature
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

# Set a threshold for the outlier detection
threshold = 1.5

# Remove the outliers from the dataset
features_no_outliers = X[~((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)]
target_no_outliers = y[~((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)]


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[24]:


for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))


# In[ ]:


df = df[~(df['PRICE'] >= 35.0)]
print(np.shape(df))


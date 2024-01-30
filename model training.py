#!/usr/bin/env python
# coding: utf-8

# # Neural network classifier
# 

# In[3]:


#get_ipython().system(' pip install tensorflow')


# In[1]:


from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
from tensorflow.keras import layers, metrics, backend

import pandas as pd
import numpy as np
import pickle


# In[2]:


np.set_printoptions(precision=3, suppress=True)


# In[3]:


# read the CSV file from the "dataset" folder
#   note: create a folder "dataset" and copy all the csv files with the network data into it
def read_data_train(dir):
  import os
  import glob

  data_train = pd.DataFrame()

  for filename in glob.glob(os.path.join(os.getcwd(), dir, "*.csv")):
    print("Loading {} ...".format(filename))
    dt = pd.read_csv(filename, sep = "\t")
    data_train = data_train.append(pd.read_csv(filename, sep = "\t"))
  
  #data_train = data_train.drop(columns = ["Forward Packets", "Forward Bytes", "Reverse Packets", "Reverse Bytes"])

  return data_train


# In[4]:


# read and shuffle the dataset
data_train = read_data_train("training")
data_train = data_train.sample(frac = 1).reset_index(drop = True)


# In[5]:


# get the label indices
labels_train = data_train.pop("Traffic Type")
labels = list(set(labels_train))
labels_train = np.array(list(map(lambda label: labels.index(label), labels_train)))

data_train = np.array(data_train)


# In[6]:


data_train.shape, labels_train


# In[7]:


# separate training and validation data
VALIDATION_COUNT = 10000
data_validation = data_train[-VALIDATION_COUNT:]
labels_validation = labels_train[-VALIDATION_COUNT:]
data_train = data_train[:-VALIDATION_COUNT]
labels_train = labels_train[:-VALIDATION_COUNT]


# In[8]:


# normalize the input
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(data_train)


# In[9]:


model = tf.keras.Sequential([
  norm_layer,
  layers.Dense(192, activation=tf.nn.relu),
  layers.Dense(384, activation=tf.nn.relu),
  layers.Dropout(0.5),
  layers.Dense(192, activation=tf.nn.relu),
  layers.Dense(12, activation=tf.nn.softmax)
])


# In[10]:


# outFun = backend.function([model.input], [norm_layer.output])
# print(data_train[0])
# print(outFun([data_train[0]]))

model.compile(
    # loss = tf.losses.MeanSquaredError(),
    loss = tf.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.optimizers.Adam(),
    #optimizer = tf.optimizers.SGD(),
    metrics = [metrics.SparseCategoricalAccuracy()]
)


# In[11]:


model.fit(data_train, labels_train, epochs = 10, batch_size = 64)


# In[14]:


print(pickle.format_version)
pickle.dump(model,open('NN','wb'))


# In[ ]:





import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from itertools import combinations

train_data=[]
train_labels=[]
test_data=[]
test_labels=[]

N=128

dataset_dir='./dataset/'
dataset_name=['xbee']

if len(sys.argv)>1:
    dataset_name[0]=sys.argv[1]

for dataset in dataset_name:
    lines=open(dataset_dir+dataset+'_normal.txt','r').readlines()
    for i,line in enumerate(lines):
        strlist=np.array(line.split())
        if i%5==0:
            test_data.append(strlist.astype(np.float))
            test_labels.append(0)         
        else:
            train_data.append(strlist.astype(np.float))
            train_labels.append(0)

for dataset in dataset_name:
    lines=open(dataset_dir+dataset+'_attack.txt','r').readlines()
    for i,line in enumerate(lines):
        strlist=np.array(line.split())
        if i%5==0:
            test_data.append(strlist.astype(np.float))
            test_labels.append(1)         
        else:
            train_data.append(strlist.astype(np.float))
            train_labels.append(1)


train_labels=np.array(train_labels)
train_data=np.array(train_data)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=N)
test_labels=np.array(test_labels)
test_data=np.array(test_data)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=N)
train_data=train_data.reshape((-1,N,1))/255.0
test_data=test_data.reshape((-1,N,1))/255.0

model = keras.Sequential()
model.add(keras.layers.Conv1D(64, 2, dilation_rate=1, activation='relu', input_shape=(N, 1)))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Conv1D(64, 2, dilation_rate=2, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Conv1D(64, 2, dilation_rate=4, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Conv1D(64, 2, dilation_rate=8, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Conv1D(64, 2, dilation_rate=16, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=100, batch_size=256, validation_data=(test_data, test_labels), verbose=1)
model.save(dataset_name[0]+'_cnn_model.h5')
model.save_weights(dataset_name[0]+'_cnn_weights.h5')
model.summary()
results = model.evaluate(test_data, test_labels)
print(results)
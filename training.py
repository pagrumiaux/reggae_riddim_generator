# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:39:22 2020

@author: PA
"""


import tensorflow as tf
from music21 import *
from dataGenerator import DataGenerator
import time
import datetime
import matplotlib.pyplot as plt
#%% train generator
paramsTrain = {'trackName': 'skank',
               'encoding' : 'one-hot',
               'batch_size': 32,
               'maxLength': 16,
               'timesteps': 8,
               'seqOverlap': 'max',
               'shuffle': True,
               'dataAugmentation': True
               }

paramsTrain['datasetPath'] = './dataset/'

trainGenerator = DataGenerator(**paramsTrain)

#%% model

# hyperparameters
timesteps = trainGenerator.timesteps
nFeat = trainGenerator.nFeatures
lstmUnits = 64
lstmAct = 'tanh'
outputUnits = nFeat
dropout = 0.2
nEpochs = 5
lr = 10e-3

# optimizer
Adam = tf.keras.optimizers.Adam(lr)

input0 = tf.keras.layers.Input(shape=(timesteps, nFeat), name='input0')
lstm0  = tf.keras.layers.LSTM(units=lstmUnits, activation=lstmAct, return_sequences = False, recurrent_dropout = dropout, name='lstm0')(input0)
output = tf.keras.layers.Dense(units=outputUnits, activation='softmax', name='output')(lstm0)

model = tf.keras.models.Model(inputs=input0, outputs=output)
print(model.summary())

model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#%% training
date = '_'+datetime.datetime.now().strftime('%Y-%m-%d')
modelPath = "lstm_64units_dataAug"
modelPath = modelPath + date + '.h5'

tic = time.time()
history = model.fit_generator(generator=trainGenerator, epochs=nEpochs, verbose=1)
model.save(filepath = modelPath)
toc = time.time()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:39:22 2020

@author: PA
"""


from tensorflow import keras
from music21 import *
from dataGenerator import DataGenerator

#%%
paramsTrain = {'trackName': 'skank',
               'encoding' : 'many-hot-close',
               'batch_size': 4,
               'timesteps': 16,
        }

paramsTrain['datasetPath'] = 'C:/Users/RQML4978/Documents/Python/music/dataset/reggae/'

trainGenerator = DataGenerator(**paramsTrain)
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:03:01 2020

@author: RQML4978
"""

import tensorflow as tf
from music21 import chord
import numpy as np
import music
#%% parameters
modelPath = 'lstm_64units_dataAug_2020-10-27.h5'
model = tf.keras.models.load_model(modelPath)

#firstChord = chord.Chord(['F4', 'A-4', 'C5'])
firstChord = music.random_chord('major')
sequenceLength = 8
timesteps = 8

#if model['encoding'] == 'one-hot':
nFeatures = 24

#%% sequence generation
encodedChordSequence = np.empty((sequenceLength, nFeatures))

inputChord = np.zeros((1, timesteps, nFeatures))
inputChord[0, -1, :] = music.encode_chord(firstChord, 'one-hot')
encodedChordSequence[0, :] = music.encode_chord(firstChord, 'one-hot')

for t in range(sequenceLength-1):
    pred = model.predict(inputChord)
    nextChordIdx = np.random.choice(np.arange(24), p=pred[0])
    nextChordEncoded = np.zeros((24))
    nextChordEncoded[nextChordIdx] = 1
    
    encodedChordSequence[t+1, :] = nextChordEncoded
    
    inputChord[:, :, :] = np.concatenate((inputChord[:, 1:, :], nextChordEncoded[np.newaxis, np.newaxis, :]), axis=1)

chordSequence = []
for c in encodedChordSequence:
    chordSequence.append(music.decode_chord(c, 'one-hot'))
    
print(chordSequence)
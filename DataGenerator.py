# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:43:08 2020

@author: PA
"""

from tensorflow import keras
from music21 import *
import os
import music
import numpy as np

#%%
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self,
               datasetPath = None,
               trackName = None,
               encoding = 'one-hot',
               batch_size = 16,
               timesteps = 4
               ):
        
        if datasetPath is None:
            raise ValueError("Please specify the absolute path of the dataset.")
        else:
            self.datasetPath = datasetPath
            
        if trackName is None:
            raise ValueError('Please specify which track you want to work with : "skank", "bass".')
        else:
            self.trackName = trackName
            
        
        # scan dataset path
        self.songList = sorted([self.datasetPath+file for file in os.listdir(self.datasetPath) if 'skank' in file and file.endswith('.mid')])
            
        # set the number of features depending on the type of encoding
        self.encoding = encoding
        if self.encoding == 'one-hot':
            self.nFeatures = 2*len(music._ROOT_NOTES)
        elif self.encoding == 'many-hot-close':
            self.nFeatures = len(music._ROOT_NOTES)
            
        # expected number of chords in the progression
        self.timesteps = timesteps
            
        self.batch_size = batch_size        
        
        self.on_epoch_end()
        
    def __len(self):
        return len(self.songList)
    
    def  __getitem__(self, idx):
        batchStart = idx * self.batch_size
        batchEnd = (idx + 1) * self.batch_size     
        batchSongs = self.songList[batchStart:batchEnd]
        inputSeq = self._data_load(batchSongs)
        
        return inputSeq#, target
        
    def on_epoch_end(self):
        return
    
    def _data_load(self, batchSongs):
        inputSeq = np.zeros((self.batch_size, self.timesteps, self.nFeatures))
        
        for iSong, song in enumerate(batchSongs):
            chordList = music.extract_chord_list(song)
            for iChord, chord in enumerate(chordList):
                inputSeq[iSong, iChord, :] = music.encode_chord(chord, self.encoding)                
        
        return inputSeq
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:43:08 2020

@author: PA
"""

from tensorflow import keras
from music21 import *
import os

class DataGenerator(keras.utils.Sequence):
    
    def __init(self,
               datasetPath = None,
               trackName = None,
               encoding = 'one-hot',
               batch_size = 16,
               ):
        
        if datasetPath is None:
            raise ValueError("Please specify the absolute path of the dataset.")
        else:
            self.datasetPath = datasetPath
            
        if track is None:
            raise ValueError('Please specify which track you want to work with : "skank", "bass".')
        else:
            self.track = track
            
        
        # scan dataset path
        songList = sorted([self.datasetPath+file for file in os.listdir(self.datasetPath) if 'skank' in file and file.endswith('.mid')])
            
        self.encoding = encoding
        
        self.batch_size = batch_size
        
        
        self.on_epoch_end()
        
    def __len(self):
        return len(songList)
    
    def  __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size        
        inpuSeq, target = self._data_load(songList[batch_start:batch_end])
        
        return inputSeq, target
        
    def on_epoch_end(self):
        return
    
    def _data_load(self):
        return
    
    
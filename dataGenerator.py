# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:43:08 2020

@author: PA
"""

from tensorflow import keras
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
               maxLength = 16,
               timesteps = 4,
               seqOverlap = 'max',
               shuffle = True,
               dataAugmentation = False
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
        self.maxLength = maxLength
        
        # training timesteps
        self.timesteps = timesteps
        if seqOverlap == 'max':
            self.seqOverlap = self.timesteps - 1
        else:
            self.seqOverlap = seqOverlap    
        self.batch_size = batch_size
        self.nSeqPerSong = int((self.maxLength-1)/(self.timesteps-self.seqOverlap))
        self.shuffle = shuffle
        self.dataAugmentation = dataAugmentation
        
        # shuffling
        self.idxSeqEpoch = None
        self.idxSongEpoch = None
        self.idxTransposeEpoch = None
        
        self.on_epoch_end()
        
    def __len__(self):
        if self.seqOverlap == 'max':
            length = len(self.songList*(self.maxLength-1)) // self.batch_size
        else:
            length = len(self.songList*self.nSeqPerSong) // self.batch_size
        
        if self.dataAugmentation:
            length = length * 12
            
        return length
    
    def  __getitem__(self, idx):
        batchStart = idx * self.batch_size
        batchEnd = (idx + 1) * self.batch_size
        
        idxSongBatch = self.idxSongEpoch[batchStart:batchEnd]
        idxSeqBatch = self.idxSeqEpoch[batchStart:batchEnd]
        idxTransposeBatch = self.idxTransposeEpoch[batchStart:batchEnd]
        
        inputSeq, target = self._data_load(idxSongBatch, idxSeqBatch, idxTransposeBatch)
        
        return inputSeq, target
        
    def on_epoch_end(self):
        nSong = len(self.songList)
        idxSeq, idxSong, idxTranspose = np.meshgrid(range(self.nSeqPerSong), range(nSong), range(12))
        idxSeq = idxSeq.flatten()
        idxSong = idxSong.flatten()
        idxTranspose = idxTranspose.flatten()
        
        indexes = np.arange(len(idxSeq))
        if self.shuffle:
            np.random.shuffle(indexes)
            
        self.idxSongEpoch = idxSong[indexes]
        self.idxSeqEpoch = idxSeq[indexes]
        self.idxTransposeEpoch = idxTranspose[indexes]
    
    def _data_load(self, idxSongBatch, idxSeqBatch, idxTransposeBatch):
        inputSeq = np.empty((self.batch_size, self.timesteps, self.nFeatures))
        target = np.empty((self.batch_size, self.nFeatures))
        
        for iSong, song in enumerate(idxSongBatch):
            songFile = self.songList[song]
            chordList = music.extract_chord_list(songFile)
            
            if self.dataAugmentation:
                chordList = music.transpose_chord_list(chordList, idxTransposeBatch[iSong])
            
            if len(chordList) < self.maxLength:
                for i in range(self.maxLength-len(chordList)):
                    chordList.append(None)
            
            # input features
            if idxSeqBatch[iSong]+1 < self.timesteps: # zero padding at the beginning of the sequence
                for t in range(self.timesteps-(idxSeqBatch[iSong]+1)):
                    inputSeq[iSong, t, :] = np.zeros((self.nFeatures))
                for t in range(idxSeqBatch[iSong]+1):
                    inputSeq[iSong, (self.timesteps-(idxSeqBatch[iSong]+1))+t, :] = music.encode_chord(chordList[t], self.encoding)
                        
            else:
                for t in range(self.timesteps): # no zero-padding needed
                    inputSeq[iSong, t, :] = music.encode_chord(chordList[idxSeqBatch[iSong]+1-self.timesteps+t], self.encoding)
            
            # target
            target[iSong, :] = music.encode_chord(chordList[idxSeqBatch[iSong]+1], self.encoding)
                    
        return inputSeq, target
    
    
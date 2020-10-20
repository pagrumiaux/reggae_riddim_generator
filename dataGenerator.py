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
        
        # shuffling
        self.idxSeqEpoch = None
        self.idxSongEpoch = None
        
        self.on_epoch_end()
        
    def __len__(self):
        if self.seqOverlap == 'max':
            return len(self.songList*(self.maxLength-1)) // self.batch_size
        else:
            return len(self.songList*self.nSeqPerSong) // self.batch_size
    
    def  __getitem__(self, idx):
        batchStart = idx * self.batch_size
        batchEnd = (idx + 1) * self.batch_size
        
        idxSongBatch = self.idxSongEpoch[batchStart:batchEnd]
        idxSeqBatch = self.idxSeqEpoch[batchStart:batchEnd]
        
        inputSeq, target = self._data_load(idxSongBatch, idxSeqBatch)
        
        return inputSeq, target
        
    def on_epoch_end(self):
        nSong = len(self.songList)
        idxSeq, idxSong = np.meshgrid(range(self.nSeqPerSong), range(nSong))
        idxSeq = idxSeq.flatten()
        idxSong = idxSong.flatten()
        
        indexes = np.arange(len(idxSeq))
        if self.shuffle:
            np.random.shuffle(indexes)
            
        self.idxSongEpoch = idxSong[indexes]
        self.idxSeqEpoch = idxSeq[indexes]
    
    def _data_load(self, idxSongBatch, idxSeqBatch):
        inputSeq = np.empty((self.batch_size, self.timesteps, self.nFeatures))
        target = np.empty((self.batch_size, self.nFeatures))
        
        for iSong, song in enumerate(idxSongBatch):
            songFile = self.songList[song]
            chordList = music.extract_chord_list(songFile)
            
            if len(chordList) < self.maxLength:
                for i in range(self.maxLength-len(chordList)):
                    chordList.append(None)
            
            # input features
            if idxSeqBatch[iSong]+1 < self.timesteps: # zero padding at the beginning of the sequence
                for t in range(self.timesteps-(idxSeqBatch[iSong]+1)):
                    inputSeq[iSong, t, :] = np.zeros((self.nFeatures))
                for t in range(idxSeqBatch[iSong]+1):
                    inputSeq[iSong, (self.timesteps-(idxSeqBatch[iSong]+1))+t, :] = music.encode_chord(chordList[t], self.encoding)
            
#            elif idxSeqBatch[iSong] + self.timesteps > self.maxLength: #zero-padding at the end of the sequence
#                for t in range(self.maxLength - idxSeqBatch[iSong]):
#                    print(t, idxSeqBatch[iSong]+1-self.timesteps+t)
#                    inputSeq[iSong, t, :] = music.encode_chord(chordList[idxSeqBatch[iSong]+1-self.timesteps+t], self.encoding)
#                for t in range(idxSeqBatch[iSong] + self.timesteps - self.maxLength):
#                    print(t, self.maxLength - idxSeqBatch[iSong]+t)
#                    inputSeq[iSong, self.maxLength - idxSeqBatch[iSong]+t, :] = np.zeros((self.nFeatures))
            
            else:
                for t in range(self.timesteps): # no zero-padding needed
                    inputSeq[iSong, t, :] = music.encode_chord(chordList[idxSeqBatch[iSong]+1-self.timesteps+t], self.encoding)

            # target
            target[iSong, :] = music.encode_chord(chordList[idxSeqBatch[iSong]+1], self.encoding)
                    
        return inputSeq, target
    
    
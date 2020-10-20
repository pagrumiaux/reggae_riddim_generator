# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:23:28 2020

@author: RQML4978
"""

import os
from music21 import *
from music import extract_chord_list
#%%
datasetPath = "./dataset/"

#%% number of files
skankFiles = [f for f in os.listdir(datasetPath) if f.endswith('skank.mid')]
bassFiles = [f for f in os.listdir(datasetPath) if f.endswith('bass.mid')]

print(f"There are {len(skankFiles)} skank files and {len(bassFiles)} bass files in this dataset.")

#%% sequence lengths
# skank
maxSequenceLength = 32
skankLengths = [0]*(maxSequenceLength+1)
for f in skankFiles:
    chordList = extract_chord_list(datasetPath + f)
    skankLengths[len(chordList)] += 1

for iLength in range(maxSequenceLength):
    if skankLengths[iLength] > 0:
        print(f"There are {skankLengths[iLength]} skank sequences of length {iLength}")
        
# bass
#bassLengths = [0]*(maxSequenceLength+1)
#for f in bassFiles:
#    chordList = extract_chord_list(datasetPath + f)
#    bassLengths[len(chordList)] += 1
#
#for iLength in range(maxSequenceLength):
#    if bassLengths[iLength] > 0:
#        print(f"There are {bassLengths[iLength]} bass sequences of length {iLength}")
        
#%%
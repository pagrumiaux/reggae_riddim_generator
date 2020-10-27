# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:11:12 2020

@author: RQML4978
"""

from music21 import converter, chord
import numpy as np

_ROOT_NOTES = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']

def extract_chord_list(skank_midi_file):
    # assume skank midi format
    score = converter.parse(skank_midi_file)
    part = score.parts[0]
    chordList = []
    for e in part:
        if isinstance(e, chord.Chord):
            chordList.append(e)
            
    return chordList

def encode_chord(c, encoding):
    if c == None:
        if encoding == 'one-hot':
            return np.zeros((len(_ROOT_NOTES)*2))
        elif encoding == 'many-hot-close':
            return np.zeros((len(_ROOT_NOTES)))
        
    else:
        c = c.simplifyEnharmonics() # use to provide a logical chord to music21
        if c.root().name in _ROOT_NOTES:
            root = c.root().name
        else:
            for p in [pitch.name for pitch in c.root().getAllCommonEnharmonics()]:
                if p in _ROOT_NOTES:
                    root = p
                    
        if encoding == 'one-hot':
            nFeatures = len(_ROOT_NOTES)*2 # minor and major are counted
            vec = np.zeros(nFeatures)
            if c.quality == 'minor':
                note_index = _ROOT_NOTES.index(root)
                vec[note_index] = 1
            elif c.quality == 'major':
                note_index = _ROOT_NOTES.index(root)
                vec[note_index+12] = 1
                
        elif encoding == 'many-hot-close':
            nFeatures = len(_ROOT_NOTES)
            vec = np.zeros(nFeatures)
            for pitch in c.pitches:
                if pitch.name not in _ROOT_NOTES:
                    for p in [enh_pitch.name for enh_pitch in pitch.getAllCommonEnharmonics()]:
                        if p in _ROOT_NOTES:
                            pitch = p
                            
                note_index = _ROOT_NOTES.index(pitch.name)
                vec[note_index] = 1
                
        return vec
    
def decode_chord(encodedChord, encoding):
    if np.count_nonzero(encodedChord) == 0:
        return None
    
    if encoding == 'one-hot':
        chordIdx = np.argmax(encodedChord)
        if chordIdx < 12:
            quality = 'minor'
        else:
            quality = 'major'
            chordIdx = chordIdx - 12
            
        root = _ROOT_NOTES[chordIdx]
        if quality == 'minor':
            third = _ROOT_NOTES[(chordIdx+3)%12]
        elif quality == 'major':
            third = _ROOT_NOTES[(chordIdx+4)%12]
            
        fifth = _ROOT_NOTES[(chordIdx+7)%12]
        
        return chord.Chord([root, third, fifth])
    
    elif encoding == 'many-hot-close':
        idxNotes = np.where(encodedChord)[0]
        chordNotes = []
        for i in idxNotes:
            chordNotes.append(_ROOT_NOTES[i])
        
        return chord.Chord(chordNotes)
        
    
def transpose_chord_list(chordList, transpositionInterval):
    chordList_transposed = []
    for c in chordList:
        chordList_transposed.append(c.transpose(transpositionInterval))
        
    return chordList_transposed

def random_chord(quality):
    randomRoot = np.random.choice(np.arange(12))
    root = _ROOT_NOTES[randomRoot]
    fifth = _ROOT_NOTES[(randomRoot+7)%12]
    if quality == 'minor':
        third = _ROOT_NOTES[(randomRoot+3)%12]
    if quality == 'major':
        third = _ROOT_NOTES[(randomRoot+4)%12]
        
    return chord.Chord([root, third, fifth])
from scikits.audiolab import wavread, wavwrite
import librosa
import numpy as np 
from os import listdir
from random import randint

class audiofile(object):
    """
    A simple container for an audio file object, containing:
    - the original data of the audio file
    - a mono version of the audio data
    - sample rate
    - encoding type
    """
    def __init__(self, data, mono, sr, enc):
        self.data = data
        self.mono = mono
        self.sr = sr
        self.enc = enc

def tomono(data):
    '''
    This may be unnecessary, but due to my use of 96kHz files and
    their poor support in librosa, this is an alternative approach
    to librosa.core.to_mono() -- note that I am only keeping the 
    left channel rather than an average.
    '''
    if type(data[0]) == np.ndarray:
        return np.array([data[i][0] for i in xrange(len(data))])
    else:
        return data

def readfiles(files):
    '''
    Given an array of file paths, reads all of the files and
    generates a list of audilfile objects as defined above.
    '''
    allfiles = []
    for f in files:
        data, sr, enc = wavread(f)
        allfiles.append(audiofile(data, tomono(data), sr, enc))
    return allfiles


def getfiles(directory):
    '''
    Given a directory, outputs a list of wav files in that
    directory.
    '''
    files = []
    for filename in listdir(directory):
        if filename.endswith(".wav"):
            files.append(filename)
    return files

def onset(af):
    # oenv = librosa.onset.onset_strength(y = af.mono, sr = af.sr)
    frames = librosa.onset.onset_detect(y = af.mono, sr = af.sr, units = 'samples', backtrack = True)
    ranges = [[frames[i], frames[i + 1] - 1] for i in xrange(len(frames) - 1)]
    ranges.append([frames[len(frames) - 1], len(af.mono) - 1])
    return ranges

def shuffle(l):
    for i in xrange(len(l)):
        k = randint(0, i)
        temp = l[k]
        l[k] = l[i]
        l[i] = temp
    return l

def collect(ranges, af):
    data = []
    for r in ranges:
        for datum in af.data[r[0]:r[1]]:
            data.append(datum)
    return audiofile(np.array(data), None, af.sr, af.enc)

def write(filename, af):
    wavwrite(af.data, filename, fs=af.sr, enc=af.enc)
    return

def rearrange(directory):
    files = getfiles(directory)
    data = readfiles(files)
    for i, f in enumerate(data):
        ranges = onset(f)
        print ranges
        shuffle(ranges)
        write(str(i) + '.wav', collect(ranges, f))




from scikits.audiolab import wavread, wavwrite
import numpy as np 
from os import listdir
from random import randint, random

class audiofile(object):
    """
    A simple container for an audio file object, containing:
    - the original data of the audio file
    - sample rate
    - encoding type
    """
    def __init__(self, data, sr, enc, name=None):
        self.data = data
        self.sr = sr
        self.enc = enc
        self.name = name

def lenSec(clip):
    '''
    Utility function: displays the length of a clip in seconds.
    '''
    return len(clip.data) / float(clip.sr)

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

def readfiles(files):
    '''
    Given an array of file paths, reads all of the files and
    generates a list of audilfile objects as defined above.
    '''
    allfiles = []
    for f in files:
        data, sr, enc = wavread(f)
        allfiles.append(audiofile(data, sr, enc, f[:-4]))
    return allfiles

def fades(lensamps):
    '''
    Computes sqrt fade envelopes of length 'lensamps' samples.
    '''
    fadein = np.arange(lensamps) / np.float(lensamps)
    fadeout = np.sqrt(np.float(1.) - fadein)
    return np.sqrt(fadein), fadeout

def fadein(clip, fade):
    '''
    Applies a fade envelope to the beginning of the clip.
    Warning: mutable operation, will operate in-place on a clip
    '''
    for i in xrange(len(fade)):
        clip[i] *= fade[i]
    return

def fadeout(clip, fade):
    '''
    Applies a fade envelope to the end of the clip.
    Warning: mutable operation, will operate in-place on a clip
    '''
    lc = len(clip)
    lf = len(fade)
    for i in xrange(lf):
        clip[lc - lf + i] *= fade[i]
    return

def genfade(clip1, clip2, samplen):
    '''
    Client function to compute a cross-fade between two clips
    '''
    tail = clip1[-samplen:]
    head = clip2[:samplen]
    fin, fout = fades(samplen)
    fadein(head, fin)
    fadeout(tail, fout)
    return head + tail

def crossfade(clip1, clip2, lenseconds):
    '''
    '''
    # ensure the sampling rates & encodings match
    assert(clip1.enc == clip2.enc and clip1.sr == clip2.sr)
    lensamps = int(lenseconds * clip1.sr) # number of samples for the cross-fade
    # ensure that the cross-fade does not exceed the sample length of the clips
    assert(lensamps < len(clip1.data) and lensamps < len(clip2.data))
    fade = genfade(clip1.data, clip2.data, lensamps)
    head = clip1.data[:-lensamps] # beginning-to-crossfade of clip 1
    tail = clip2.data[lensamps:]  # crossfade-to-end of clip 2
    # head -> crossfade -> tail, reshape is necessary due to np.append flattening the array:
    result = np.append(np.append(head, fade), tail).reshape(-1, 2)
    # combine resulting clip into an audio file:
    return audiofile(result, clip1.sr, clip1.enc)

def combine(clip1, clip2, gapseconds):
    '''
    '''
    # ensure the sampling rates & encodings match
    assert(clip1.enc == clip2.enc and clip1.sr == clip2.sr)
    gapsamps = int(gapseconds * clip1.sr)
    # since we are combining stereo files, we can force the array of correct size (pad with zeros):
    gap = np.zeros(gapsamps, 2)
    result = np.append(np.append(clip1.data, gap), clip2.data).reshape(-1, 2)
    return audiofile(result, clip1.sr, clip1.enc)

def trimLeft(clip):
    '''
    Returns the index to the left of the first non-zero item in an audio buffer
    '''
    counter = 1
    while (counter != len(clip)):
        if clip[counter][0] >= 1e-4 or clip[counter][1] >= 1e-4:
            return counter - 1
        counter += 1
    return counter - 1

def trimRight(clip):
    '''
    Returns the index to the right of the last non-zero item in an audio buffer
    '''
    counter = len(clip) - 2
    while (counter != -1):
        if abs(clip[counter][0]) >= 1e-4 or abs(clip[counter][1]) >= 1e-4:
            return counter + 1
        counter -= 1
    return 0

def trim(clip):
    '''
    '''
    clip.data = clip.data[trimLeft(clip.data) : trimRight(clip.data)]
    return

def trimAll(files):
    '''
    '''
    for f in files:
        trim(f)
    return

def declick(clip):
    '''
    Because some files have an intro click, we need to:
    1. cut off some small amount of an audio file (8 ms) from both sides
    2. fade in / out the edges
    '''
    eightms = int(clip.sr * .008)    # eight ms cut
    fadesamps = int(clip.sr * 0.005) # five ms fade
    fin, fout = fades(fadesamps)
    clip.data = clip.data[eightms : -eightms]
    fadein(clip.data, fin)
    fadeout(clip.data, fout)
    return

def prerpareclip(clip):
    '''
    Utility function: preparatory steps to 
    '''
    trim(clip)
    declick(clip)
    return

def shuffle(l):
    for i in xrange(len(l)):
        k = randint(0, i)
        temp = l[k]
        l[k] = l[i]
        l[i] = temp
    return

def trackify(path, filename):
    clips = readfiles(getfiles(path))
    indices = range(len(clips))
    lengths = []
    for c in clips:
        lengths.append(lenSec(c))
    maxFade = min(lengths)
    minFade = maxFade / 3
    shuffle(indices)
    result = crossfade(clips[indices[0]], clips[indices[1]], random.random() * (maxFade - minFade) + minFade)
    for i in xrange(2, len(indices)):
        maxFade = lenSec(clips[indices[i]])
        result = crossfade(result, clips[indices[i]], random.random() * (.666 * maxFade) + (.333 * maxFade))
    wavwrite(result.data, filename, result.sr, result.enc)
    return

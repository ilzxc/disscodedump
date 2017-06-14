import librosa
import numpy as np
import essentia
from essentia.standard import Envelope
from scikits.audiolab import wavread, wavwrite
from os import listdir

def getfiles(directory = '.'):
    '''
    Given a directory, outputs a list of wav files in that
    directory.
    '''
    if directory[-1] != '/':
        directory += '/'
    files = []
    for filename in listdir(directory):
        if filename.endswith(".wav"):
            files.append(directory + filename)
    return files

def process(file):
    # read in the file
    f, sr, enc = wavread(file)
    # compute the fourier transform & compute the window times:
    D = librosa.stft(f)
    times = librosa.frames_to_samples(np.arange(D.shape[1]))
    # compute the onset strength envelope:
    env = librosa.onset.onset_strength(y=f, sr=sr)
    assert(len(times) == len(env))
    # compute the onsets we are actually interested in, convert to samples:
    onsets = librosa.onset.onset_detect(y=f, sr=sr)
    onset_samps = librosa.frames_to_samples(onsets)
    assert(onset_samps[-1] <= len(f))
    # create a lookup table for retrieving onset strenghts:
    lookup = []
    prevval = 0
    for v in onset_samps:
        for i in xrange(prevval, len(times)):
            if times[i] == v:
                lookup.append(i)
                prevval = i + 1
                break
    # create an empty audio buffer (result):
    result = np.zeros(len(f))
    # write envelope onset strength values at every onset point
    # computed by the envelope:
    for i in xrange(len(lookup)):
        result[onset_samps[i]] = env[lookup[i]]
    # write the result:
    wavwrite(result, file[:-4] + '_proc.wav', sr, enc)
    return

def envelopefile(file, attack = 1, release = 10):
    # read in the file:
    f, sr, enc = wavread(file)
    env = Envelope()
    env.configure(attackTime = attack, releaseTime = release)
    result = env(essentia.array(f))
    wavwrite(result, file[:-4] + '_env.wav', sr, enc)
    return

def envelope(file, attack = 1, release = 10):
    # read in the file:
    f, sr, enc = wavread(file)
    env = Envelope()
    env.configure(attackTime = attack, releaseTime = release)
    result = env(essentia.array(f))
    # wavwrite(result, file[:-4] + '_env.wav', sr, enc)
    return result.reshape(-1)

def clicktrack(file):
    # read in the file:
    f, sr, enc = wavread(file)
    env = Envelope()
    env.configure(attackTime = 0, releaseTime = 5)
    curve = env(essentia.array(f))
    result = np.zeros(len(curve))
    i = 1
    while i < len(curve):
        if curve[i] - curve[i - 1] > 0.05:
            result[i] = curve[i] # record the click at the onset
            i += 1100 # advance the playhead by 1100 samples (~22 ms) to avoid closely-spaced clicks
            continue
        i += 1
    wavwrite(result, file[:-4] + '_clicks.wav', sr, enc)
    return

def clicks(file):
    # read in the file:
    f, sr, enc = wavread(file)
    env = Envelope()
    env.configure(attackTime = 0, releaseTime = 5)
    curve = env(essentia.array(f))
    result = []
    i = 1
    while i < len(curve):
        if curve[i] - curve[i - 1] > 0.05:
            result.append([i, curve[i]])
            i += 1100 # advance the playhead by 1100 samples (~22 ms) to avoid closely-spaced clicks
            continue
        i += 1
    return result, len(f)

'''a numpy-optimized "sum at index" function'''
def insert(L, i, s): L[i : i + len(s)] += s

def clicke(length):
    short = np.zeros(int(44100 * (length / 1000.)))
    short[0] = 1.0
    decay = length / 6
    env = Envelope()
    env.configure(attackTime = 0, releaseTime = decay)
    return env(essentia.array(short))

def loadfile(file):
    f, sr, enc = wavread(file)
    return essentia.array(f), sr, enc

def transients(file, transients, decay):
    '''
    articulates attacks indicated by the onset extraction using short-decay
    samples of the 
    '''
    idx, l = clicks(file)
    result = np.zeros(l * 2) # multiply by 2 for stereo
    ce = clicke(decay)
    for i in idx:
        # read in a random transient:
        trans, sr, enc = wavread(transients[random.randint(0, len(transients) - 1)])
        # assume stereo, extract subset to length of ce
        trans = trans.reshape(-1)                 # flatten the array
        tl = trans[0 : len(ce) * 2     : 2] * ce  # extract left & multiply by the envelope
        tr = trans[1 : len(ce) * 2 + 1 : 2] * ce  # extract right & multiple by the envelope
        fin = np.vstack((tl, tr)).reshape((-1,), order='F')
        # scale the result according to onset strength:
        fin *= i[1]
        insert(result, i[0] * 2, fin)
    return result.reshape(-1, 2)

def resonances(file, resonances):
    '''
    stitches together a file of sustained resonances by looking at boundaries
    defined by the onset extraction -- r
    '''
    idx, l = clicks(file)
    idx.append([l, 0])
    env = envelope(file, 5, 20)
    result = np.zeros(l * 2)
    for i in xrange(0, len(idx) - 1):
        # read in a random resonance:
        reson, sr, enc = wavread(resonances[random.randint(0, len(resonances) - 1)])
        # assume stereo. need to extract subset equal to onset length:
        reson = reson.reshape(-1) # flatten
        length = (idx[i + 1][0] - idx[i][0]) * 2
        start = random.randint(0, len(reson) / 2 - length) # random start place
        rl = reson[0 : length : 2]
        rr = reson[1 : length + 1 : 2]
        fin = np.vstack((rl, rr)).reshape((-1,), order='F')
        insert(result, idx[i][0] * 2, fin)
    left = result[::2] * env
    right = result[1::2] * env
    return np.vstack((left, right)).reshape((-1), order='F').reshape(-1, 2)

def applyenvelope(stereo, envelope):
    assert(len(stereo[0]) == 2)
    assert(len(stereo) == len(envelope))
    stereo = stereo.reshape(-1)
    left = stereo[::2] * envelope
    right = stereo[1::2] * envelope
    return np.vstack((left, right)).reshape((-1), order='F')


























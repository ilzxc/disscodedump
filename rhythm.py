import librosa
import numpy as np
import essentia
import random
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
    runmax = 0
    while i < len(curve):
        if curve[i] - curve[i - 1] > 0.05:
            result.append([i, curve[i]])
            if curve[i] > runmax: runmax = curve[i]
            i += 1100 # advance the playhead by 1100 samples (~22 ms) to avoid closely-spaced clicks
            continue
        i += 1
    # normalize output:
    if runmax != 0:
        factor = 1. / runmax
        for entry in result:
            entry[1] *= factor
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
    samples of the glass stuff
    '''
    idx, l = clicks(file)
    if len(idx) == 0:
        return np.zeros(l * 2).reshape(-1, 2)
    result = np.zeros(l * 2 + int((decay / 1000.) * 88200) + 2) # multiply by 2 for stereo
    ce = clicke(decay)
    for i in idx:
        # read in a random transient:
        trans, sr, enc = wavread(transients[random.randint(0, len(transients) - 1)])
        # assume stereo, extract subset to length of ce
        trans = trans.reshape(-1)                 # flatten the array
        tl = trans[0 : len(trans) : 2]  # extract left & multiply by the envelope
        tr = trans[1 : len(trans) : 2]  # extract right & multiple by the envelope
        if len(tl) < len(ce):
            nl = np.zeros(len(ce))
            nr = np.zeros(len(ce))
            insert(nl, 0, tl)
            insert(nr, 0, tr)
            tl = nl * ce
            tr = nr * ce
        else:
            tl = tl[0 : len(ce)] * ce
            tr = tr[0 : len(ce)] * ce
        fin = np.vstack((tl, tr)).reshape((-1,), order='F')
        # scale the result according to onset strength:
        fin *= i[1]
        insert(result, i[0] * 2, fin)
    result *= 1. / result.max() # normalize
    return result.reshape(-1, 2)

def resonances(file, resonances, crossfadems):
    '''
    stitches together a file of sustained resonances by looking at boundaries
    defined by the onset extraction -- r
    '''
    crossfade = int((crossfadems / 1000.) * 44100) # gives us crossfade in samples
    rampup = np.sqrt(np.array([x / (crossfade - 1.) for x in xrange(0, crossfade)]))
    rampdown = rampup[::-1]
    ramplen = len(rampup) * 4
    rampstart = len(rampup) * 2
    # compute the clicks used for boundaries
    idx, l = clicks(file)
    idx = [[rampstart, 1.0]] + idx
    idx.append([l - ramplen, 0]) # add the last click so that we add the resonance for the last event

    env = envelope(file, 10, 30)
    env *= (1 / env.max()) # normalize envelope
    
    result = np.zeros(l * 2)
    for i in xrange(0, len(idx) - 1):
        # read in a random resonance:
        f = resonances[random.randint(0, len(resonances) - 1)]
        reson, sr, enc = wavread(f)
        # assume stereo. need to extract subset equal to onset length:
        reson = reson.reshape(-1) # flatten
        length = (idx[i + 1][0] - idx[i][0]) * 2 + ramplen
        # TODO: check to ensure that len(reson) / 2 > length
        rl = None
        rr = None
        if len(reson) >= length:
            start = random.randint(0, len(reson) - length) # random start place
            rl = reson[start : length + start : 2]
            rr = reson[start + 1 : length + start + 1 : 2]
        else:
            # just copy the buffer into a zero-padded array:
            print "else triggered"
            print len(reson), length, length - len(reson)
            temp = np.zeros(length)
            insert(temp, 0, reson)
            rl = temp[0::2]
            rr = temp[1::2]
            assert(len(rl) == len(rr) == length / 2)
        # print length, len(rl), len(rr)
        # fade up rl & rr
        rl[:len(rampup)] *= rampup
        rr[:len(rampup)] *= rampup
        rl[-len(rampup):] *= rampdown
        rr[-len(rampup):] *= rampdown
        # combine & insert into result
        fin = np.vstack((rl, rr)).reshape((-1,), order='F')
        insert(result, idx[i][0] * 2 - rampstart, fin)
    result *= (1 / result.max())
    left = result[::2] * env
    right = result[1::2] * env
    result = np.vstack((left, right)).reshape((-1), order='F')
    result *= (1 / result.max())
    return result.reshape(-1, 2)

def applyenvelope(stereo, envelope):
    assert(len(stereo[0]) == 2)
    assert(len(stereo) == len(envelope))
    stereo = stereo.reshape(-1)
    left = stereo[::2] * envelope
    right = stereo[1::2] * envelope
    return np.vstack((left, right)).reshape((-1), order='F')

def procFile(v, prefix):
    wavwrite(transients(v, trans, 20), prefix + v[-9:-4] + '_trans00.wav', 44100, 'pcm24')
    wavwrite(transients(v, trans, 10), prefix + v[-9:-4] + '_trans01.wav', 44100, 'pcm24')
    wavwrite(resonances(v, reson, 10), prefix + v[-9:-4] + '_reson00.wav', 44100, 'pcm24')
    wavwrite(resonances(v, reson, 15), prefix + v[-9:-4] + '_reson01.wav', 44100, 'pcm24')
    return

def procGestur(g, c, prefix):
    wavwrite(resonances(g, c, 15), prefix + g[:-9:-4] + '_reson.wav', 44100, 'pcm24')


def makeTransients(vox, prefix):
    wavwrite(transients(v, trans, 100), prefix + v[-9:-4] + '_transLong.wav', 44100, 'pcm24')
    wavwrite(transients(v, trans, 20), prefix + v[-9:-4] + '_transShort.wav', 44100, 'pcm24')
    wavwrite(transients(v, bulbs, 100), prefix + v[-9:-4] + '_bulbs.wav', 44100, 'pcm24')
    wavwrite(transients(v, tiny, 100), prefix + v[-9:-4] + '_tiny.wav', 44100, 'pcm24')



















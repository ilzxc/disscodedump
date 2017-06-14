import librosa
import random

ppq = 1000      # 1000 ticks for a half-second
q = .5          # @ 120 bpm, each quarter note is 1/2 seconds

s2t = lambda s: int((s / q) * ppq)
rv = lambda: int(random.random() * 127)

pattern = midi.Pattern()
pattern.resolution = ppq
track = midi.Track()
pattern.append(track)

def event(track, lengthSeconds, velocity=127):
    track.append(midi.NoteOnEvent(tick=0, velocity=velocity, pitch=midi.C_4))
    track.append(midi.NoteOffEvent(tick=s2t(lengthSeconds), pitch=midi.C_4))

# times = [0, .25, .49, .75, .76, .77, .90, .97, 1.]

y, sr = librosa.load('vocals_membrane.wav')
frames = librosa.onset.onset_detect(y=y,sr=sr)
times = librosa.frames_to_time(frames, sr=sr)

for i in range(0, len(times) - 1):
    eventLen = times[i + 1] - times[i]
    event(track, eventLen, rv())
track.append(midi.EndOfTrackEvent(tick=1))

print pattern

midi.write_midifile('testTimes.mid', pattern)

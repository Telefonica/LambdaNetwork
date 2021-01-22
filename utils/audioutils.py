# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:34:28 2019

@author: W. Fernando López Gavilánez


This file contains util code for audio prodessing.

Preconditions about audio format:
    Filetype: .wav
    Sampling frequency: 16000Hz
    Channels: 1
    Frame lengh: 2 bytes
    Endianness: little-endian
"""


import pyaudio
import numpy as np
import time
import wave
from random import randint
import subprocess
import os
from scipy.io import wavfile


def play_audio(filename):
    """
    To reproduce an audio file in background and waits until end its execution

    Args:
        filename: full path to the audio file
    Returns:
        0 when the proccess has finished
    """
    process = subprocess.Popen(['aplay', '-q', filename])
    return process.wait()

def play_audio_fast(filename):
    """
    To reproduce an audio file in background and waits until end its execution

    Args:
        filename: full path to the audio file
    Returns:
        process
    """
    process = subprocess.Popen(['aplay', '-q', filename])
    return process

def save_wav(audio_signal, filename, ch=1, bytes_per_sample=2, fs=16000):
    """
    To store and audio contained in a numpy array.

    Args:
        audio_signal: Numpy array conatining the audio samples
        filename: Name of the file to save, including the full path
        ch: Number of channels (1 or 2)
        bytes_per_sample: Frame length
        fs: Sampling rate
    Returns:
        none
    """
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(ch)
    wav_file.setsampwidth(bytes_per_sample)
    wav_file.setframerate(fs)
    wav_file.writeframes(audio_signal.tobytes())
    wav_file.close()


def save_wav_from_bytes(audio_signal, filename, ch=1, bytes_per_sample=2, fs=16000):
    """
    To store and audio contained in a numpy array.

    Args:
        audio_signal: Numpy array conatining the audio samples
        filename: Name of the file to save, including the full path
        ch: Number of channels (1 or 2)
        bytes_per_sample: Frame length
        fs: Sampling rate
    Returns:
        none
    """
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(ch)
    wav_file.setsampwidth(bytes_per_sample)
    wav_file.setframerate(fs)
    wav_file.writeframes(audio_signal)
    wav_file.close()


def open_wav_file(path):
    audio_wav = wave.open(path,'r')
    audio_bytes = audio_wav.readframes(-1)
    return np.frombuffer(audio_bytes, dtype='int16')


def open_wavfile(path):
    rate, data = wavfile.read(path)
    return rate, data


def open_stream_channel(fs=16000, ch=1):
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,channels=ch,rate=fs,output=True)
    return stream


def close_stream_channel(stream):
    stream.close()
    pyaudio.PyAudio().terminate()


def generate_white_noise(len, k=32):
    k = 32 #noise level, the higher the louder
    return (np.random.normal(0,1.3,len)*k).astype('int16')


def combine_signals(x, y):
    """
    Combine two signals of the same size

    Args:
        x: Audio signal 1, numpy array
        y: Audio signal 2, numpy array

    Returns:
        Combined signal
    """
    return (x + y)


def concat_signals(x,y):
    return np.append(x, y).astype('int16')

  
def stream_audio_bytes(stream, audio_bytes):
    stream.write(audio_bytes)
    time.sleep(1)


def change_signal_level(signal, factor):
    """
       Changes the signal level.
       Recommendable values: 1.3 and 0.7
    """
    return (signal*factor).astype('int16')


def change_signal_pitch(signal, factor):
    """ Multiplies the sound's speed by some `factor` 
        Recommendable values of factor: 0.9 (slower) and 1.1 (faster)
    """
    indices = np.round( np.arange(0, len(signal), factor) )
    indices = indices[indices < len(signal)].astype(int)
    return signal[ indices.astype(int) ]


def add_backgound_noise(signal, backgound_sound, index, fs):
    """
    Adds white noise and ambience noise the utterance. The user must take care of the index.
    

    Args:
        signal: Short signal to combine with ambience sound and withe noise
        backgound_sound: Long signal that will be splitted and combined with a short utterance 
        index: Index of where to start cropping the background_sound signal.
        fs: Sampling rate of both signals, to control the times.
    Retuns:
        _signal: Signal combining ambience noise and utterance
        index: New index
    """
    delay = randint(fs, 2*fs) #Delay between one and two seconds
    wn_chunk1 = generate_white_noise(delay)
    wn_chunk2 = generate_white_noise(len(signal))
    ambience_chunk = backgound_sound[index:index + delay + len(signal)]
    _signal = concat_signals(wn_chunk1, combine_signals(signal, wn_chunk2))
    _signal = combine_signals(_signal, ambience_chunk)
    index = index + delay + len(_signal)
    return (_signal, index)


def startRecording():
    cmd = ['arecord', '-q', '-t', 'wav', '-c', '1', '-f', 's16', '-r', '16000', 'tmp.wav']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return process


def stopRecording(process):
    process.terminate()
    process.stdout.close()


def setFormat(n, path, wake_word):
    """
    Adapts an audio file to the following format:
        Filetype: .wav
        Sampling frequency: 16000Hz
        Channels: 1
        Frame lengh: 2 bytes
        Endianness: little-endian

    The file should be named tmp.wav and should be in the execution path

    Args:
        n: Number to name the resultant audio.
        path: Path to leave the result
        wake_word: String to name the resultant audio
    Returns:
        None
    """
    os.system('ffmpeg -y -i tmp.wav -ar 16000 -ac 1 ' + path + '/' + wake_word + ".{0:0=2d}".format(n) + ".wav")
    os.system('rm tmp.wav')


def trim_silence(audio_signal, silence_threshold):
    """
    Trim the silence of an audio signal. Considers silence samples which level is under a given threshold

    Args:
        audio_signal: Audio signal as numpy array
        silence_threshold: Threshold to determine silence or not
    Returns:
        Trimed signal and its lengh as a tuple
    """
    offset = 1000 #Offset to crop 62.5ms
    start = 0 + int(5*offset)
    end = len(audio_signal) - 1 - offset
    for i in range(start, len(audio_signal)):
        if (audio_signal[i] > silence_threshold):
            start = i - 1 - int(offset/2)
            break
    for i in range(len(audio_signal)):
        if (audio_signal[end - i] > silence_threshold):
            end = len(audio_signal) - i + 1 + int(offset/2)
            break
    return (audio_signal[start:end], len(audio_signal[start:end]))


def trim_synthetic_silence(audio_signal):
    start = 0
    end = len(audio_signal) - 1
    for i in range(len(audio_signal)):
        if (audio_signal[i] != 0):
            start = i - 1
            break
    for i in range(len(audio_signal)):
        if (audio_signal[end - i] != 0):
            end = len(audio_signal) - i + 1
            break
    return (audio_signal[start:end], len(audio_signal[start:end]))


def change_signal_level(signal, factor):
    """
       Changes the signal level.
       Recommendable values: 1.3 and 0.7
    """
    return (signal*factor).astype('int16')


def change_signal_pitch(signal, factor):
    """ 
    Multiplies the sound's speed by some factor
    Recommendable values of factor: 0.9 (slower) and 1.1 (faster)
    """
    indices = np.round( np.arange(0, len(signal), factor) )
    indices = indices[indices < len(signal)].astype(int)
    return signal[ indices.astype(int) ]


def get_utterance_custom(client, text, voice, pitch, rate, volume, duration):
    """
    Uses the Microsoft cognitive service to get a custom utterance

    Args:
        client: MSFT TTS Client
        text: text to convert into speech
        voice: voice os the speech
        picth: pitch ['x-low', 'low', 'medium', 'high', 'x-high', 'default']
        rate: velocity of teh speech ['x-low', 'slow', 'medium', 'fast', 'x-fast', 'default']
        volume: volume of the speech ['x-soft', 'soft', 'medium', 'loud', 'x-loud', 'default']
        duartion: duration of the speech ['500ms', '600ms', '700ms', '800ms', '900ms', '1s', '1100ms', '1200ms', '1300ms', '1400ms']
    Returns:
        numpy array with the synthetic utterance
    """
    UTTERANCE_START = 44 #RIFF format
    utterance_bytes = client.say_custom(text, voice=voice, pitch=pitch, rate=rate, volume=volume, duration=duration)
    utterance = np.frombuffer(utterance_bytes, dtype='int16')
    utterance_len = len(utterance)
    return (utterance[UTTERANCE_START:], utterance_len)

class Recorder(object):

    """
    Creates a recorder object which allows an audio input stream. 
    The input stream can be passed in the constructor, if not 
    it will be used created a new input stream

    """

    def __init__(self, stream=None):
        self._done = False
        self._chunk_size = 2048

        if(stream != None):
            self.stream = stream
            self._closeStream = False
        else:
            self._pa = pyaudio.PyAudio()
            self._closeStream = True
            self.stream = self._pa .open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=16000,
                                                input=True,
                                                frames_per_buffer=self._chunk_size)       

    def record(self):
        while not self._done:
            data = self.stream.read(self._chunk_size)
            yield data

    def get_chunk(self):
        return self.stream.read(self._chunk_size, exception_on_overflow=False)

    def getChunkDuration(self):
        return self._chunk_size/16000

    def done(self):
        self._done = True
        if(self._closeStream):
            self.stream.close()
            self._pa.terminate() 
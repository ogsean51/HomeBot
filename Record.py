import pyaudio
from queue import Queue
from threading import Thread
import sys
import time
import numpy as np
import matplotlib.mlab as mlab
import numpy as np
import time
from pydub import AudioSegment
import sys
from scipy.io.wavfile import write
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import LiteProcessor as lp
import wave
import audioop
from datetime import datetime

#Setting to true prints more information, uses timeout
debugMode = True

#chunk_duration = 3 # Each read length in seconds from mic.
#fs = 48000 # sampling rate for mic
#chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
log_data = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 3
COUNT = 0


feed_duration = 3
feed_samples = int(RATE * feed_duration)

#assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

# Queue to communiate between the audio callback and main thread
q = Queue()

run = True

#the higher this is the higher the threshold
silence_threshold = 120

# Run the demo for a timeout seconds
timeout = time.time() + 0.25*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

rms = None


def get_spectrogram(wav):
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
    '''nfft = 300 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx'''

def input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=0,
        stream_callback=callback)
    return stream


def process(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold, debugMode, rms
    if debugMode:
        if time.time() > timeout:
            run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data,data0)
    
    rms = audioop.rms(in_data, 2)
    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

def initialize(debug):
    global data, q, timeout, run, debugMode, COUNT, log_data
    print("Initializing...")
    debugMode = debug
    stream = input_stream(process)
    stream.start_stream()
    print("---------------------- INITIALIZED ----------------------")

    try:
        while run:
            data = q.get()
            print(data.shape)
            
            if len(log_data) > 0:
                log_data = log_data + data
            else:
                log_data = data
                
            #prediction = lp.process(data)
            
            #print(prediction)
            
            #if debugMode:
                #print(prediction)
    except (KeyboardInterrupt, SystemExit):
        print("SystemExit")
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False
        
    print("\n ---------------------- STOPPING ----------------------")
    
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    file = "./Logs/Audio/log.wav"
    
    with wave.open(file, 'wb') as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(pyaudio.get_sample_size(FORMAT))
        out.setframerate(RATE)
        out.writeframes(b''.join(log_data))
    out.close()
    
    stream.stop_stream()
    stream.close()
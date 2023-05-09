import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
import os

interpreter = None
input_details = None
output_details = None

'''#loads file from path, utilized by preprocess()
#returns wav data
def load_wav(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
'''
#preprocesses file loaded from path
#returns spectrogram
def preprocess(input):
    wav = tf.convert_to_tensor(input)
    #wav, sample_rate = tf.audio.decode_wav(data_tensor, desired_channels=1)

    
    wav = tf.cast(wav, dtype=tf.float32)
    print(wav.shape)
    #wav = tf.squeeze(wav, axis=)

    wav = tfio.audio.resample(wav, rate_in=44100, rate_out=16000)
    
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    output = np.asarray(spectrogram, dtype=np.float32)
    
    return output

def initialize():
    global interpreter, input_details, output_details

    interpreter = tf.lite.Interpreter(model_path="./Training/Models/processor.tflite")

    input_details = interpreter.get_input_details()    
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    #input_shape = input_details[0]['shape']

def process(data):
    global interpreter, input_details, output_details
    audio = preprocess(data)

    input_data = np.asarray(audio, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    #Predict model with processed data  
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])  

    return prediction


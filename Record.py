import pyaudio
import wave
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 0.5
WAVE_OUTPUT_FILENAME = "./Process-Segments/"
device_index = 2

INDEX = 0
BOTTOM = 0


def record():
    #print("recording via index " + str(index))
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=0,
                        frames_per_buffer=CHUNK)
    Recordframes = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    file = WAVE_OUTPUT_FILENAME + "analyze" + str(INDEX) + ".wav"
    INDEX += 1
    if(INDEX > 4):
        INDEX = 0
    waveFile = wave.open(file, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close() 

def reset():
    os.remove("./Process-Segments/analyze" + str(BOTTOM) + ".wav")
    BOTTOM += 1
    
    if(BOTTOM > 4):
        BOTTOM = 0
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "./Data/WakeWord/"
COMMAND = "Homebot"
device_index = 2

NUMBEROFFILES = 100


def record(count):
    #print("recording via index " + str(index))
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=0,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME + COMMAND + "/" + COMMAND + str(count) + ".wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()



count = 0
for i in range(0, NUMBEROFFILES):
    record(count)
    count += 1
    print(count)
print("DONE")
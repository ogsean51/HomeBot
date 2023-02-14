import WakeProcessor as wp
import Record as r
import os
import sounddevice as sd


def wake():
    print("WAKE")
    
    #for f in os.listdir("./Process-Segments"):
        #os.remove("./Process-Segments/" + f)
    print("CLEARED")

def main():
    print("initializing...")
    model = wp.initialize()
    print("initialized \n-------------------")
    while True:
        print("running...")
        r.record(10)
        if(wp.process(model)):
            print("WAKE")
        r.reset()
        print("reset \n-------------------")

if __name__ == "__main__":
    main()
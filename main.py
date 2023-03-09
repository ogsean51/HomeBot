import WakeProcessor as wp
import Record as r
import os

def wake():
    print("WAKE")
    
    #for f in os.listdir("./Process-Segments"):
        #os.remove("./Process-Segments/" + f)
    print("CLEARED")

def main():
    print("initializing...")
    #model = wp.initialize()
    print("initialized \n-------------------")
    while True:
        print("running...")
        r.record()
        if(wp.process()):
            print("WAKE")
        r.reset()
        print("reset \n-------------------")

if __name__ == "__main__":
    main()
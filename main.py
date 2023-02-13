import WakeProcessor as wp
import Record as r

def main():
    print("running...")
    model = wp.initialize()
    
    while True:
        r.record()
        if(wp.process(model)):
            print("WAKE")
        r.reset()

if __name__ == "__main__":
    main()
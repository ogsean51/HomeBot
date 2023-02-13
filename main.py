import WakeProcessor as wp

def main():
    print("running...")
    model = wp.initialize()
    wp.process(model)

if __name__ == "__main__":
    main()
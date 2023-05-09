import LiteProcessor as lp
import Record as r
from multiprocessing import Process
import os
from art import *                                           

def debugMode():
    debug = input("Run in debug mode? y/n  -  ")
    if debug == "y" or debug =="Y":
        return True
    if debug == "n" or debug =="N":
        return False


def wake():
    print("WAKE")
    
    #for f in os.listdir("./Process-Segments"):
    #   os.remove("./Process-Segments/" + f)
    print("CLEARED")

def main():
    print(text2art('''HOME BOT''', font="big"))
    lp.initialize()
    r.initialize(debugMode())

if __name__ == "__main__":
    main()
import WakeProcessor as wp
import Record as r
from multiprocessing import Process
import os

header =''' _   _  _____         ___       ___    _____  _____ 
( ) ( )(  _  )/'\_/`\(  _`\    (  _`\ (  _  )(_   _)
| |_| || ( ) ||     || (_(_)   | (_) )| ( ) |  | |  
|  _  || | | || (_) ||  _)_    |  _ <'| | | |  | |  
| | | || (_) || | | || (_( )   | (_) )| (_) |  | |  
(_) (_)(_____)(_) (_)(____/'   (____/'(_____)  (_)  
                                                    
                                                    '''
                                                    

def debugMode():
    debug = input("Run in debug mode? y/n         ")
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
    print(header)
    #model = wp.initialize()
    r.initialize(debugMode())

if __name__ == "__main__":
    main()
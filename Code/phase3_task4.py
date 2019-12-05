import helpers
import numpy as np
import PPR
import SVM
import DTree

def main():
    clf=int(input("Select your preferred classifier model: \n1. SVM\n2. Decision-Tree\n3. PPR\t:\t"))
    if clf==1:
        SVM.SVM()
    elif clf==2:
        DTree.DTC()
    elif clf==3:
        PPR.PPR()
    else:
        print("Error!! Invalid input")
        exit(1)

if __name__=="__main__":
    main()

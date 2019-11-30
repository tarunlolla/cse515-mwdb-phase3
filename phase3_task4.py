import helpers
import numpy as np

def SVM():
    print("SVM")
    pass

def DTree():
    print("DTree")
    pass

def PPR():
    print("PPR")
    featureVectors,image_list=helpers.getFeatureVectors()
    latentVectors=helpers.computePCA(featureVectors,20)
    pass

def main():
    #un_dataset=input("Enter the path for unlabelled images: ")
    #clf=int(input("Select your preferred classifier model: \n1. SVM\n2. Decision-Tree\n3. PPR\t:\t"))
    clf=3
    if clf==1:
        SVM()
    elif clf==2:
        DTree()
    elif clf==3:
        PPR()
    else:
        print("Error!! Invalid input")
        exit(1)
    un_dataset='/home/tarunlolla/MWDB/Phase3/phase3_sample_data/Unlabelled/Set 1'
    un_dataset_md='/home/tarunlolla/MWDB/Phase3/phase3_sample_data/unlabelled_set1.csv'

if __name__=="__main__":
    main()

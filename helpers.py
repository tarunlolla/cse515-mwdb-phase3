import os
import pandas as pd
from sklearn.decomposition import PCA
import connectToDB
import numpy as np
from scipy.spatial import distance

def fetchDatasetDetails():
    dataset_input=open("dataset.txt",'r')
    dataset_lines=dataset_input.readlines()
    dataset_dict=dict()
    for line in dataset_lines:
        a = line.rstrip().split('=')
        dataset_dict[a[0]] = a[1]
    dataset_input.close()
    return dataset_dict['dataset_path'],dataset_dict['metadata_file']

def computePCA(dataMatrix,k):
    if(k!=0):
        pca = PCA(n_components=k)
    else:
        pca = PCA(n_components=None)
    # fits the model with matrix and applies dimensionality reduction on the matrix
    trans_matrix = pca.fit_transform(dataMatrix)
    return trans_matrix

def getFeatureVectors():
    db = connectToDB.connectToDB()
    collection = db.HandInfo
    all_images = []
    dataMatrix=[]
    records=collection.find({},{"imageName":1,"_id":0}).sort("imageName")
    for record in records:
        all_images.append(record['imageName'])
    for img in all_images:
        fm=connectToDB.getFeatureVectorDB(db,'hog',img)
        dataMatrix.append(fm)
    dataMatrix = np.array(dataMatrix)
    return dataMatrix, all_images

def findEuclideanDistance(vector1,vector2):
    return distance.euclidean(vector1,vector2)

def make_html(folder, image, extras):
    text='<div style="height:175px;width:175px;border:1px;border-style:solid;border-color:rgb(0,0,0);"><img src="{}" style="display:block;height:120px;width:160px;margin:5px;"/><p style="font: italic smaller sans-serif; text-align:center;">{}<br>{}</p></div>'.format(os.path.join(folder, image),image,extras)
    #print(text)
    return text

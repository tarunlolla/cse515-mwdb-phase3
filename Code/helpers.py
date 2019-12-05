import os
import pandas as pd
from sklearn.decomposition import PCA
import connectToDB
import numpy as np
from scipy.spatial import distance

"""
ll_dataset_path=/home/tarunlolla/MWDB/Phase3/cse515-mwdb-phase3.git/phase3_sample_data/Labelled/Set2
ll_metadata_file=/home/tarunlolla/MWDB/Phase3/cse515-mwdb-phase3.git/phase3_sample_data/labelled_set2.csv
ull_dataset_path=/home/tarunlolla/MWDB/Phase3/cse515-mwdb-phase3.git/phase3_sample_data/Unlabelled/Set2
ull_metadata_file=/home/tarunlolla/MWDB/Phase3/cse515-mwdb-phase3.git/phase3_sample_data/unlabelled_set2.csv
master_metadata_file=/home/tarunlolla/MWDB/Phase3/cse515-mwdb-phase3.git/phase3_sample_data/HandInfo.csv
"""

def fetchDatasetDetails(task_name='initial'):
    if task_name=='task3' or task_name=='initial':
        dataset_input=open("dataset.txt",'r')
        dataset_lines=dataset_input.readlines()
        dataset_dict=dict()
        for line in dataset_lines:
            a = line.rstrip().split('=')
            dataset_dict[a[0]] = a[1]
        dataset_input.close()
        return dataset_dict['dataset_path'],dataset_dict['metadata_file']
    elif task_name=='task4':
        dataset_input=open("dataset_"+str(task_name)+".txt",'r')
        dataset_lines=dataset_input.readlines()
        dataset_dict=dict()
        for line in dataset_lines:
            a = line.rstrip().split('=')
            dataset_dict[a[0]] = a[1]
        dataset_input.close()
        return dataset_dict['ll_dataset_path'],dataset_dict['ll_metadata_file'],dataset_dict['ull_dataset_path'],dataset_dict['ull_metadata_file'],dataset_dict['master_metadata_file']

def computePCA(dataMatrix,k):
    if(k!=0):
        pca = PCA(n_components=k)
    else:
        pca = PCA(n_components=None)
    # fits the model with matrix and applies dimensionality reduction on the matrix
    trans_matrix = pca.fit_transform(dataMatrix)
    return trans_matrix

def findEuclideanDistance(vector1,vector2):
    return 1-distance.cosine(vector1,vector2)

def make_html(folder, image, extras):
    text='<div style="height:175px;width:175px;border:1px;border-style:solid;border-color:rgb(0,0,0);"><img src="{}" style="display:block;height:120px;width:160px;margin:5px;"/><p style="font: italic smaller sans-serif; text-align:center;">{}<br>{}</p></div>'.format(os.path.join(folder, image),image,extras)
    #print(text)
    return text

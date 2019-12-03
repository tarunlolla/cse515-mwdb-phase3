import os
import pandas as pd
from sklearn.decomposition import PCA

import connectToDB
import helpers
import numpy as np
import webbrowser as wb

def getSimilarK(vectors,img_list,k,n):
    img_dict=dict()
    img_dist_list=[]
    for i in range(0,len(vectors)):
        dist=helpers.findEuclideanDistance(vectors[i],vectors[n])
        img_dist_list.append((img_list[i],dist))
    return_list=sorted((img_dist_list),key=lambda x: x[1])
    return return_list[1:int(k)+1]

def buildSimGraph(vectors,img_list,k):
    graphDict=dict()
    for img in img_list:
        graphDict[img]=[]
    for i in range(len(list(vectors))):
        graphDict[img_list[i]]=getSimilarK(vectors,img_list,k,i)
    return graphDict

def buildTMatrix(simGraph_Dict,img_list):
    rowsInT=[]
    for i in range(0,len(img_list)):
        image=img_list[i]
        row=[0]*len(img_list)
        img_sim_list=[x[0] for x in simGraph_Dict[image]]
        img_sim_val_list=[x[1] for x in simGraph_Dict[image]]
        p=sum(img_sim_val_list)
        for img in img_list:
            if img in img_sim_list:
                idx=img_sim_list.index(img)
                row[idx]=img_sim_val_list[idx]/p
        rowsInT.append(row)
    return np.array(rowsInT)


def PPR(simGraph_Dict,img_id1,img_id2,img_id3,img_list):
    tp_vector=[]
    a=0.85
    for img in img_list:
        if img in [img_id1,img_id2,img_id3]:
            tp_vector.append(1/3)
        else:
            tp_vector.append(0)
    T=buildTMatrix(simGraph_Dict,img_list)
#    print(T)
    I=np.identity(len(img_list),float)
    tp_vector=np.array(tp_vector)
    # print(simGraph_Dict)
    #print(T.shape,I.shape,np.transpose(tp_vector).shape)
    if a==1:
        pi=np.linalg.inv(np.subtract(I,(a*T)))
    else:
        pi=np.matmul(np.linalg.inv(np.subtract(I,(a*T))),(1-a)*tp_vector)
    #print(pi.shape)
    return list(pi)

def visualise(images):
    dataset_path,metadata_path=helpers.fetchDatasetDetails('task3')
    files=[x[0] for x in images]
    ppr=[x[1] for x in images]
    text=''
    for i in range(0,len(files)):
        text += helpers.make_html(dataset_path, files[i],ppr[i])
    html_file=open('render_task3.html','w')
    #print(dataset_path,metadata_path)
    html_file.write('<div style="display: grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(8, 5vw);grid-gap: 100px;">'+text+'</div>')
    wb.open_new_tab("render_task3.html")

def getFeatureVectors():
    db = connectToDB.connectToDB()
    dataset_path,metadata_path=helpers.fetchDatasetDetails('task3')
    dataMatrix=[]
    all_images = os.listdir(dataset_path)
    for image in all_images:
        vec=connectToDB.getFeatureVectorDB(db,'hog',image)
        dataMatrix.append(vec)
    #collection = db.HandInfo
    #records=collection.find({},{"imageName":1,"_id":0}).sort("imageName")
    # for record in records:
    #     all_images.append(record['imageName'])
    # for img in all_images:
    #     fm=connectToDB.getFeatureVectorDB(db,'hog',img)
    #     dataMatrix.append(fm)
    dataMatrix = np.array(dataMatrix)
    return dataMatrix, all_images

def main():
    featureVectors,image_list=getFeatureVectors()
    latentVectors=helpers.computePCA(featureVectors,20)
    # k=input("Enter the value of k :")
    k=5
    simGraph_Dict=buildSimGraph(latentVectors,image_list,k)
    # img_id1=input("Enter Image ID 1: ")
    # img_id2=input("Enter Image ID 2: ")
    # img_id3=input("Enter Image ID 3: ")
    # K=input("Enter the value of K :")
    #Hand_0008333.jpg, Hand_0006183.jpg, Hand_0000074.jpg
    # img_id1='Hand_0008333.jpg'
    # img_id2='Hand_0006183.jpg'
    # img_id3='Hand_0000074.jpg'
    #Hand_0003457.jpg,Hand_0000074.jpg, Hand_0005661.jpg
    img_id1='Hand_0003457.jpg'
    img_id2='Hand_0000074.jpg'
    img_id3='Hand_0005661.jpg'
    K=10
    #print(simGraph_Dict[img_id1][:K])
    #print(simGraph_Dict[img_id2][:K])
    #print(simGraph_Dict[img_id3][:K])
    ppr=PPR(simGraph_Dict,img_id1,img_id2,img_id3,image_list)
    dominantImages=[]
    ppr_temp=ppr
    img_list_temp=image_list
    for i in range(0,K):
        max_val=max(ppr_temp)
        idx=ppr_temp.index(max_val)
        dominantImages.append((img_list_temp[idx],ppr_temp[idx]))
        ppr_temp.remove(max_val)
        img_list_temp.remove(img_list_temp[idx])
    for x in dominantImages:
        print(x)
    visualise(dominantImages)
    #print(ppr)

if __name__ == '__main__':
    main()

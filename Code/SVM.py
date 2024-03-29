import connectToDB
import helpers
import pandas as pd
import numpy as np
import cv2
import initialSteps
from scipy.spatial import distance
import webbrowser as wb
import matplotlib.pyplot as plt

def getFeatureVectorsLL(dataset,metadata):
    imageList=list(metadata['imageName'])
    aOh=list(metadata['aspectOfHand'])
    imgVector=[]
    for i in range(0,len(imageList)):
        image=imageList[i]
        aspectOfHand=aOh[i]
        print(dataset+'/'+image)
        img=cv2.imread(dataset+'/'+image)
        hog=initialSteps.compute_HOG(img)
        imgVector.append([image,hog,aspectOfHand])
    feature_df=pd.DataFrame(imgVector,columns=['imageName','featureVector','aspectOfHand'])
    #print(feature_df)
    latentVectors=helpers.computePCA(np.array(list(feature_df['featureVector'])),2)
    #print(latentVectors)
    #latentVectors=list(feature_df['featureVector'])
    print("latentVectors computed")
    feature_df['latentVector']=list(latentVectors)
    return feature_df

def getFeatureVectorsULL(dataset,metadata):
    imageList=list(metadata['imageName'])
    imgVector=[]
    for i in range(0,len(imageList)):
        image=imageList[i]
        print(dataset+'/'+image)
        img=cv2.imread(dataset+'/'+image)
        hog=initialSteps.compute_HOG(img)
        imgVector.append([image,hog])
    feature_df=pd.DataFrame(imgVector,columns=['imageName','featureVector'])
    #print(feature_df)
    latentVectors=helpers.computePCA(np.array(list(feature_df['featureVector'])),2)
    # print(latentVectors)
    #latentVectors=list(feature_df['featureVector'])
    print("latentVectors computed")
    feature_df['latentVector']=list(latentVectors)
    return feature_df


def computeDistance(df1,df2):
    distance_df=pd.DataFrame(columns=['dorsal_img','palmar_img','distance'])
    for i in range(df1.shape[0]):
        for j in range(df2.shape[0]):
            d=distance.cosine(df1['latentVector'][i],df2['latentVector'][j])
            distance_df=distance_df.append({'dorsal_img':df1['imageName'][i],'palmar_img':df2['imageName'][j],'distance':round(d,2)},ignore_index=True)
    return distance_df

def nearestToOrigin(v1,v2):
    print(v1)
    print(v2)
    org=[0]*20
    d1=distance.cosine(org,v1)
    d2=distance.cosine(org,v2)
    print(d1,d2)
    if d1>d2:
        return 1
    else:
        return 0

def SVClassify(ull_feature_df,w,b,dm,pm):
    aspects=[]
    for i in range(0,ull_feature_df.shape[0]):
        vec=np.array(ull_feature_df['latentVector'][i])
        c=np.dot(vec,w)+b
        if c<0:
            aspects.append('palmar')
        elif c>=0:
            aspects.append('dorsal')
    print(ull_feature_df.shape)
    print(len(aspects))
    ull_feature_df['aspectOfHand']=list(aspects)
    return ull_feature_df

def visualise(df,ds):
    images=list(df['imageName'])
    aspect=list(df['aspectOfHand'])
    text=''
    for i in range(0,len(images)):
        text += helpers.make_html(ds, images[i],aspect[i])
    html_file=open('render_task4.html','w')
    # print(dataset_path,metadata_path)
    html_file.write('<div style="display: grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(8, 5vw);grid-gap: 100px;">'+text+'</div>')
    wb.open_new_tab("render_task4.html")

def SVM():
    lDim=2
    lbl_ds,lbl_md,ull_ds,ull_md,master_md=helpers.fetchDatasetDetails('task4')
    lbl_md_df=pd.read_csv(lbl_md,delimiter=',')
    lbl_md_df=lbl_md_df.replace({'dorsal left': 'dorsal','dorsal right': 'dorsal','palmar left': 'palmar','palmar right': 'palmar'})
    lbl_fv=[]
    for img in list(lbl_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        lbl_fv.append([img,vec])
    lbl_feature_df=pd.DataFrame(lbl_fv,columns=['imageName','featureVector'])
    lbl_feature_df['aspectOfHand']=list(lbl_md_df['aspectOfHand'])
    latentVectors=helpers.computePCA(np.array(list(lbl_feature_df['featureVector'])),lDim)
    print("latentVectors computed")
    lbl_feature_df['latentVector']=list(latentVectors)
    # lbl_feature_df=getFeatureVectorsLL(lbl_ds,lbl_md_df)
    ull_md_df=pd.read_csv(ull_md,delimiter=',')
    ull_fv=[]
    for img in list(ull_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        ull_fv.append([img,vec])
    ull_feature_df=pd.DataFrame(ull_fv,columns=['imageName','featureVector'])
    latentVectors=helpers.computePCA(np.array(list(ull_feature_df['featureVector'])),lDim)
    print("latentVectors computed")
    ull_feature_df['latentVector']=list(latentVectors)
    # ull_feature_df=getFeatureVectorsULL(ull_ds,ull_md_df)
    master_md_df=pd.read_csv(master_md,delimiter=',')
    #Dividing dataframes for Dorsal and Palmar
    cond=lbl_feature_df['aspectOfHand']=='palmar'
    rows = lbl_feature_df.loc[cond, :]
    palmarDF = pd.DataFrame(columns=lbl_feature_df.columns)
    palmarDF = palmarDF.append(rows, ignore_index=True)
    dorsalDF = lbl_feature_df
    dorsalDF.drop(rows.index, inplace=True)
    if lDim==2:
        for i in range(len(list(dorsalDF['latentVector']))):
            plt.scatter(list(dorsalDF['latentVector'])[i][0],list(dorsalDF['latentVector'])[i][1],label='Dorsal',c='red',s=80)
        for i in range(len(list(palmarDF['latentVector']))):
            plt.scatter(list(palmarDF['latentVector'])[i][0],list(palmarDF['latentVector'])[i][1],label='Palmar',c='blue',s=80)
    distance_df=computeDistance(dorsalDF,palmarDF)
    #print(distance_df.head(10))
    print(distance_df['distance'].min())
    minPairIDX=list(distance_df['distance']).index(distance_df['distance'].min())
    print(minPairIDX,distance_df.loc[[minPairIDX]])
    idx_dorsal=list(dorsalDF.loc[dorsalDF['imageName'] == distance_df['dorsal_img'][minPairIDX]].index)[0]
    idx_palmar=list(palmarDF.loc[palmarDF['imageName'] == distance_df['palmar_img'][minPairIDX]].index)[0]
    print(dorsalDF.loc[[idx_dorsal]])
    print(palmarDF.loc[[idx_palmar]])
    dorsal_margin=dorsalDF['latentVector'][idx_dorsal]
    palmar_margin=palmarDF['latentVector'][idx_palmar]
    if lDim==2:
        plt.scatter(dorsal_margin[0],dorsal_margin[1],marker='+',c='red',s=80)
        plt.scatter(palmar_margin[0],palmar_margin[1],marker='+',c='blue',s=80)
        plt.show()
    midPoint=[]
    for i in range(0,len(dorsal_margin)):
        midPoint.append((dorsal_margin[i]+palmar_margin[i])/2)
    b=(np.dot(np.array(midPoint),np.array(midPoint).transpose()))*(-1)
    dorsal_margin_val=np.dot(np.array(midPoint),np.array(dorsal_margin))+b
    palmar_margin_val=np.dot(np.array(midPoint),np.array(palmar_margin))+b
    print(dorsal_margin_val,palmar_margin_val)
    classified_DF=SVClassify(ull_feature_df,np.array(midPoint),b,dorsal_margin_val,palmar_margin_val)
    visualise(classified_DF,ull_ds)
    master_md_images=list(master_md_df['imageName'])
    master_md_aspects=list(master_md_df['aspectOfHand'])
    for img in master_md_images:
        if img not in list(ull_feature_df['imageName']):
            idx=master_md_images.index(img)
            master_md_images.remove(img)
            master_md_aspects.remove(master_md_aspects[idx])
    cl_images=list(classified_DF['imageName'])
    cl_aspects=list(classified_DF['aspectOfHand'])
    sum=0.0
    for i in range(0,len(cl_images)):
        img=cl_images[i]
        idx=master_md_images.index(img)
        master_label=master_md_aspects[idx].replace(" right","").replace(" left","")
        print(img,master_label,cl_aspects[i])
        if master_label==cl_aspects[i]:
            sum += 1
    print("Accuracy = "+str(sum/len(cl_images)))

if __name__=='__main__':
    SVM()

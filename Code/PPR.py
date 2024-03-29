import helpers
import pandas as pd
import connectToDB
import phase3_task3 as t3
import numpy as np
import webbrowser as wb

def trainPPR(df,val):
    img_list=list(df['imageName'])
    graphDict=t3.buildSimGraph(df['featureVector'],list(df['imageName']),5)
    T=t3.buildTMatrix(graphDict,img_list)
    a=0.15
    I=np.identity(len(img_list),float)
    tp_prob=len(img_list)-val
    tp_vector=[1/tp_prob]*val + [0]*tp_prob
    tp_vector=np.array(tp_vector)
    pi=np.matmul(np.linalg.inv(np.subtract(I,(a*T))),(1-a)*tp_vector)
    #print(len(ppr))
    return pi

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

def PPR():
    lbl_ds,lbl_md,ull_ds,ull_md,master_md=helpers.fetchDatasetDetails('task4')
    lbl_md_df=pd.read_csv(lbl_md,delimiter=',')
    lbl_md_df=lbl_md_df.replace({'dorsal left': 'dorsal','dorsal right': 'dorsal','palmar left': 'palmar','palmar right': 'palmar'})
    lbl_fv=[]
    for img in list(lbl_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        lbl_fv.append([img,vec])
    lbl_feature_df=pd.DataFrame(lbl_fv,columns=['imageName','featureVector'])
    lbl_feature_df['aspectOfHand']=list(lbl_md_df['aspectOfHand'])
    #print(lbl_feature_df)
    cond=lbl_feature_df['aspectOfHand']=='palmar'
    rows = lbl_feature_df.loc[cond, :]
    palmarDF = pd.DataFrame(columns=lbl_feature_df.columns)
    palmarDF = palmarDF.append(rows, ignore_index=True)
    dorsalDF = lbl_feature_df
    dorsalDF.drop(rows.index, inplace=True)
    ull_md_df=pd.read_csv(ull_md,delimiter=',')
    master_md_df=pd.read_csv(master_md,delimiter=',')
    ull_fv=[]
    for img in list(ull_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        ull_fv.append([img,vec])
    ull_feature_df=pd.DataFrame(ull_fv,columns=['imageName','featureVector'])
    print("Training the model")
    #print(lbl_feature_df.shape)
    #ppr_ll=trainPPR(lbl_feature_df)
    nanList=[np.nan]*ull_feature_df.shape[0]
    ull_feature_df['aspectOfHand']=nanList
    val1=dorsalDF.shape[0]
    val2=palmarDF.shape[0]
    print("Value = "+str(val1)+" "+str(val2))
    ddf_ull=dorsalDF.append(ull_feature_df,ignore_index=True)
    pdf_ull=palmarDF.append(ull_feature_df,ignore_index=True)
    ddf_ppr=trainPPR(ddf_ull,val1)
    pdf_ppr=trainPPR(pdf_ull,val2)
    ddf_ppr=ddf_ppr[val1:]
    pdf_ppr=pdf_ppr[val2:]
    for i in range(0,len(ddf_ppr)):
        if ddf_ppr[i]>pdf_ppr[i]:
            ull_feature_df['aspectOfHand'][i]='dorsal'
        else:
            ull_feature_df['aspectOfHand'][i]='palmar'
    classified_DF=ull_feature_df
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
    PPR()

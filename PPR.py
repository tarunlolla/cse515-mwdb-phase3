import helpers
import pandas as pd
import connectToDB
import phase3_task3 as t3

def main():
    lbl_ds,lbl_md,ull_ds,ull_md,master_md=helpers.fetchDatasetDetails('task4')
    lbl_md_df=pd.read_csv(lbl_md,delimiter=',')
    lbl_md_df=lbl_md_df.replace({'dorsal left': 'dorsal','dorsal right': 'dorsal','palmar left': 'palmar','palmar right': 'palmar'})
    lbl_fv=[]
    for img in list(lbl_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        lbl_fv.append([img,vec])
    lbl_feature_df=pd.DataFrame(lbl_fv,columns=['imageName','featureVector'])
    lbl_feature_df['aspectOfHand']=list(lbl_md_df['aspectOfHand'])
    print(lbl_feature_df)
    ull_md_df=pd.read_csv(ull_md,delimiter=',')
    master_md_df=pd.read_csv(master_md,delimiter=',')
    ull_fv=[]
    for img in list(ull_md_df['imageName']):
        vec=connectToDB.getFeatureVectorDB(connectToDB.connectToDB(),'hog',img)
        ull_fv.append([img,vec])
    ull_feature_df=pd.DataFrame(ull_fv,columns=['imageName','featureVector'])
    


if __name__=='__main__':
    main()

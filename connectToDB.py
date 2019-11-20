import pymongo

def connectToDB():
    conn_file=open("db_details.txt",'r')
    lines=conn_file.readlines()
    db_details=dict()
    for line in lines:
        a=line.rstrip().split('=')
        db_details[a[0]] = a[1]
    conn_file.close()
    conn = pymongo.MongoClient(db_details['db_host'],int(db_details['db_port']))
    db=conn[db_details['db_sid']]
    return db

def insertLBPtoDB(db,imgName,featureVector):
    fv_1=featureVector[0:len(featureVector)//2]
    fv_2=featureVector[len(featureVector)//2:len(featureVector)]
    print(len(fv_1))
    print(len(fv_2))
    collection1=db.lbp1
    collection2=db.lbp2
    collection1.insert_one({'_id':str(imgName) , 'featureVector' : fv_1 })
    collection2.insert_one({'_id':str(imgName) , 'featureVector' : fv_2 })

def getLBPfeatureDB(db,imgName):
    collection1=db.lbp1
    collection2=db.lbp2
    for record in collection1.find({'_id' : str(imgName) }):
        fv1=record['featureVector']
    for record in collection2.find({'_id' : str(imgName) }):
        fv2=record['featureVector']
    return fv1+fv2

def getFeatureVectorDB(db,feature,imgName):
    if feature == 'cm':
        collection=db.cm
        for record in collection.find({'_id' : str(imgName) }):
            featureVector=record['featureVector']
        return featureVector
    elif feature == 'sift':
        collection=db.sift
        for record in collection.find({'_id' : str(imgName) }):
            featureVector=record['featureVector']
        return featureVector
    elif feature=='hog':
        collection=db.hog
        for record in collection.find({'_id' : str(imgName) }):
            featureVector=record['featureVector']
        return featureVector
    elif feature=='lbp':
        return getLBPfeatureDB(db,imgName)

def insertFeatureToDB(db,feature,imgName,featureVector):
    if db == '':
        db=connectToDB()

    if feature == 'cm':
        collection=db.cm
        collection.insert_one({'_id':str(imgName),'featureVector':featureVector})
    elif feature == 'sift':
        collection=db.sift
        collection.insert_one({'_id':str(imgName),'featureVector':featureVector})
    elif feature=='hog':
        collection=db.hog
        collection.insert_one({'_id':str(imgName),'featureVector':featureVector})
    elif feature=='lbp':
        insertLBPtoDB(db,imgName,featureVector)


#def insertLStoDB(db,featureModel,):


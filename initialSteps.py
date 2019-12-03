import connectToDB
import os
from progressbar import ProgressBar,RotatingMarker,Bar,ETA,Counter
from skimage import feature
import cv2
import pandas as pd
import helpers

def compute_HOG(image):
    scale_percent = 10  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)  # to resize the image
    feature_output, hog_image = feature.hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                            block_norm='L2-Hys', visualize=True)
    return list(feature_output)

def loadMetadata(db,md_file):
    collection=db.HandInfo
    md=pd.read_csv(md_file)
    collection.insert_many(md.to_dict('records'))

def main():
    db=connectToDB.connectToDB()
    dataset_path,metadata_path=helpers.fetchDatasetDetails()
    print("Loading metadata file")
    loadMetadata(db,metadata_path)
    print("Reading images at : "+dataset_path)
    images=os.listdir(dataset_path)
    widgets = ['Files Processed: ', Counter(), ' ', Bar(marker=RotatingMarker()), ETA()]
    pbar=ProgressBar(widgets=widgets)
    for image in pbar(images):
        img=cv2.imread(dataset_path+'/'+image)
        hog=compute_HOG(img)
        #print('Writing computed vectors to Database for image ' + image)
        connectToDB.insertFeatureToDB(db,'hog',image,list(hog))

if __name__ == '__main__':
    main()


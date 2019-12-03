from __future__ import print_function
import helpers
import numpy as np
import cv2
import initialSteps
import pandas as pd
import webbrowser as wb
def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])
def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
#######
# class_counts(training_data)
#######
def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)
#######
class Question:
    """A Question is used to partition a dataset.
    """
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return float(val) >= float(self.value)
        else:
            return val == self.value
    #def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
    #    condition = "=="
    #    if is_numeric(self.value):
    #        condition = ">="
    #    return "Is %s %s?" % (condition, str(self.value))
#######
def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
#######
def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
#######
def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
#######
def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value

    and calculating the information gain."""

    best_gain = 0  # keep track of the best information gain

    best_question = None  # keep train of the feature / value that produced it

    current_uncertainty = gini(rows)

    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column
        for val in values:  # for each value
            question = Question(col, val)
            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)
            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain > best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

#######
class Leaf:
    """A Leaf node classifies data.
    """
    def __init__(self, rows):

        self.predictions = class_counts(rows)

class Decision_Node:

    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self,

                 question,

                 true_branch,

                 false_branch):

        self.question = question

        self.true_branch = true_branch

        self.false_branch = false_branch

def build_tree(rows):
    """Builds the tree."""
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)
    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)
    # Recursively build the true branch.
    true_branch = build_tree(true_rows)
    # Recursively build the false branch.
    false_branch = build_tree(false_rows)
    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on the answer.
    return Decision_Node(question, true_branch, false_branch)
def print_tree(node, spacing=""):
    # Base case: we've reached a leaf

    if isinstance(node, Leaf):

        print (spacing + "Predict", node.predictions)

        return
    # Print the question at this node
    print (spacing + str(node.question))
    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)
#######
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs
#######
def getFeatureVectorsLL(dataset,metadata):
    imageList=list(metadata['imageName'])
    aOh=list(metadata['aspectOfHand'])
    imgVector=[]
    for i in range(0,len(imageList)):
        image=imageList[i]
        aspectOfHand=aOh[i]
        img=cv2.imread(dataset+'/'+image)
        hog=initialSteps.compute_HOG(img)
        imgVector.append([image,hog,aspectOfHand])
    feature_df=pd.DataFrame(imgVector,columns=['imageName','featureVector','aspectOfHand'])
    latentVectors=helpers.computePCA(np.array(list(feature_df['featureVector'])),60)
    #print(latentVectors)
    #latentVectors=list(feature_df['featureVector'])
    #print("latentVectors computed")
    feature_df['latentVector']=list(latentVectors)
    return feature_df
def getFeatureVectorsULL(dataset,metadata):
    imageList=list(metadata['imageName'])
    imgVector=[]
    for i in range(0,len(imageList)):
        image=imageList[i]
        img=cv2.imread(dataset+'/'+image)
        hog=initialSteps.compute_HOG(img)
        imgVector.append([image,hog])

    feature_df=pd.DataFrame(imgVector,columns=['imageName','featureVector'])
    latentVectors=helpers.computePCA(np.array(list(feature_df['featureVector'])),60)

    # print(latentVectors)

    #latentVectors=list(feature_df['featureVector'])

    #print("latentVectors computed")

    feature_df['latentVector']=list(latentVectors)

    return feature_df
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
def DTC():
    DecTreeVector=[]
    unaspects=[]
    lbl_ds,lbl_md,ull_ds,ull_md,master_md=helpers.fetchDatasetDetails('task4')
    lbl_md_df=pd.read_csv(lbl_md,delimiter=',')
    lbl_md_df=lbl_md_df.replace({'dorsal left': 'dorsal','dorsal right': 'dorsal','palmar left': 'palmar','palmar right': 'palmar'})
    lbl_feature_df=getFeatureVectorsLL(lbl_ds,lbl_md_df)
    for i in range(len(lbl_feature_df['latentVector'])):
        DecTreeVector.append(np.append(lbl_feature_df['latentVector'].loc[i],lbl_feature_df['aspectOfHand'].loc[i]))
    ull_md_df=pd.read_csv(ull_md,delimiter=',')
    ull_feature_df=getFeatureVectorsULL(ull_ds,ull_md_df)
    master_md_df=pd.read_csv(master_md,delimiter=',')
    my_tree = build_tree(DecTreeVector)
    #print_tree(my_tree)
    for data in ull_feature_df['latentVector']:
        predicted=classify(data, my_tree)
        print ("Predicted:%s" %predicted)   #Actual: %s.  %(row[-1]
        for key in predicted.keys():
            unaspects.append(key)
    ull_feature_df['aspectOfHand']=list(unaspects)
    visualise(ull_feature_df,ull_ds)
    cl_images=list(ull_feature_df['imageName'])
    cl_aspects=list(ull_feature_df['aspectOfHand'])
    master_md_images=list(master_md_df['imageName'])
    master_md_aspects=list(master_md_df['aspectOfHand'])
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
    DTC()

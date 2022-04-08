from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
# import opencv
from skimage.io import imread
from skimage.transform import resize
import time
import sys
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

overAllTimeStart = time.time()

loop_time = 20
acc = np.zeros([5, loop_time])
tp = np.zeros([5, loop_time])
tn = np.zeros([5, loop_time])
fp = np.zeros([5, loop_time])
fn = np.zeros([5, loop_time])
avg_time = np.zeros([5, loop_time])


def load_image_files(container_path, dimension=(224, 224, 3)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
        print(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    # print(images)
    return Bunch(data=flat_data, target=target, target_names=categories,
                 images=images, DESCR=descr), folders


image_dataset_train, folders_train = load_image_files("../covid-chestxray-dataset/output/train/")
image_dataset_test, folders_test = load_image_files("../covid-chestxray-dataset/output/test/")

# image_dataset_train, folders_train = load_image_files("train/")
# image_dataset_test, folders_test = load_image_files("test/")

# split image
X_train = image_dataset_train.data
y_train = image_dataset_train.target

X_test = image_dataset_test.data
y_test = image_dataset_test.target
print(len(X_train), len(X_test))

for loop in range(loop_time):
    print(loop, "out of", loop_time)

    ########################################SVM###############################################
    start = time.time()

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    end = time.time()
    # print("SVM_time", end - start)

    y_pred_SVM = clf.predict(X_test)

    # print(y_pred)
    # print(y_test)

    len_of_y = len(y_pred_SVM)
    predict_correct_covid = 0
    predict_wrong_covid = 0
    predict_correct_noncovid = 0
    predict_wrong_noncovid = 0

    for i in range(len_of_y):
        if y_pred_SVM[i] == y_test[i] and y_pred_SVM[i] == 0:
            predict_correct_covid += 1
        elif y_pred_SVM[i] == y_test[i] and y_pred_SVM[i] == 1:
            predict_correct_noncovid += 1
        elif y_pred_SVM[i] != y_test[i] and y_pred_SVM[i] == 0:
            predict_wrong_covid += 1
        elif y_pred_SVM[i] != y_test[i] and y_pred_SVM[i] == 1:
            predict_wrong_noncovid += 1

    # print("predict_correct_covid", predict_correct_covid)
    # print("predict_wrong_covid", predict_wrong_covid)
    # print("predict_correct_noncovid", predict_correct_noncovid)
    # print("predict_wrong_noncovid", predict_wrong_noncovid)
    # print("percen of correct covid", predict_correct_covid / (predict_correct_covid + predict_wrong_covid))
    # print("percen of correct noncovid", predict_correct_noncovid / (predict_correct_noncovid + predict_wrong_noncovid))
    # print("precent over all", (predict_correct_covid + predict_correct_noncovid) / len_of_y)
    # print("---------------------------------------------------------------------------------------------------\n")

    acc[0][loop] = (predict_correct_covid + predict_correct_noncovid) / len_of_y
    tp[0][loop] = predict_correct_covid
    tn[0][loop] = predict_correct_noncovid
    fp[0][loop] = predict_wrong_covid
    fn[0][loop] = predict_wrong_noncovid
    avg_time[0][loop] = end - start

    ##################################Bayesian########################################################
    start = time.time()

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    end = time.time()
    # print("Naive Bayesian train time", end - start)

    y_pred_NB = gnb.predict(X_test)

    # print(y_pred_NB)
    # print(y_test)

    len_of_y = len(y_pred_NB)
    predict_correct_covid = 0
    predict_wrong_covid = 0
    predict_correct_noncovid = 0
    predict_wrong_noncovid = 0

    for i in range(len_of_y):
        if y_pred_NB[i] == y_test[i] and y_pred_NB[i] == 0:
            predict_correct_covid += 1
        elif y_pred_NB[i] == y_test[i] and y_pred_NB[i] == 1:
            predict_correct_noncovid += 1
        elif y_pred_NB[i] != y_test[i] and y_pred_NB[i] == 0:
            predict_wrong_covid += 1
        elif y_pred_NB[i] != y_test[i] and y_pred_NB[i] == 1:
            predict_wrong_noncovid += 1

    # print("predict_correct_covid", predict_correct_covid)
    # print("predict_wrong_covid", predict_wrong_covid)
    # print("predict_correct_noncovid", predict_correct_noncovid)
    # print("predict_wrong_noncovid", predict_wrong_noncovid)
    # print("percen of correct covid", predict_correct_covid/(predict_correct_covid + predict_wrong_covid))
    # print("percen of correct noncovid", predict_correct_noncovid/(predict_correct_noncovid + predict_wrong_noncovid))
    # print("precent over all", (predict_correct_covid + predict_correct_noncovid)/len_of_y)
    # print("---------------------------------------------------------------------------------------------------\n")

    acc[1][loop] = (predict_correct_covid + predict_correct_noncovid) / len_of_y
    tp[1][loop] = predict_correct_covid
    tn[1][loop] = predict_correct_noncovid
    fp[1][loop] = predict_wrong_covid
    fn[1][loop] = predict_wrong_noncovid
    avg_time[1][loop] = end - start

    ##################################DecisionTree########################################################
    start = time.time()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    end = time.time()
    # print("decision tree train time", end - start)

    y_pred_DS = clf.predict(X_test)

    # print(y_pred)
    # print(y_test)

    len_of_y = len(y_pred_DS)
    predict_correct_covid = 0
    predict_wrong_covid = 0
    predict_correct_noncovid = 0
    predict_wrong_noncovid = 0

    for i in range(len_of_y):
        if y_pred_DS[i] == y_test[i] and y_pred_DS[i] == 0:
            predict_correct_covid += 1
        elif y_pred_DS[i] == y_test[i] and y_pred_DS[i] == 1:
            predict_correct_noncovid += 1
        elif y_pred_DS[i] != y_test[i] and y_pred_DS[i] == 0:
            predict_wrong_covid += 1
        elif y_pred_DS[i] != y_test[i] and y_pred_DS[i] == 1:
            predict_wrong_noncovid += 1

    # print("predict_correct_covid", predict_correct_covid)
    # print("predict_wrong_covid", predict_wrong_covid)
    # print("predict_correct_noncovid", predict_correct_noncovid)
    # print("predict_wrong_noncovid", predict_wrong_noncovid)
    # print("percen of correct covid", predict_correct_covid/(predict_correct_covid + predict_wrong_covid))
    # print("percen of correct noncovid", predict_correct_noncovid/(predict_correct_noncovid + predict_wrong_noncovid))
    # print("precent over all", (predict_correct_covid + predict_correct_noncovid)/len_of_y)
    # print("---------------------------------------------------------------------------------------------------\n")

    acc[2][loop] = (predict_correct_covid + predict_correct_noncovid) / len_of_y
    tp[2][loop] = predict_correct_covid
    tn[2][loop] = predict_correct_noncovid
    fp[2][loop] = predict_wrong_covid
    fn[2][loop] = predict_wrong_noncovid
    avg_time[2][loop] = end - start

    ##################################RandomForest########################################################

    start = time.time()

    clf = RandomForestClassifier(max_depth=10)
    clf.fit(X_train, y_train)

    end = time.time()
    # print("RandomForest train time", end - start)

    y_pred_RF = clf.predict(X_test)
    y_pred_prob_RF = clf.predict_proba(X_test)

    # print(y_pred)
    # print(y_test)
    len_of_y = len(y_pred_RF)
    predict_correct_covid = 0
    predict_wrong_covid = 0
    predict_correct_noncovid = 0
    predict_wrong_noncovid = 0

    for i in range(len_of_y):
        if y_pred_RF[i] == y_test[i] and y_pred_RF[i] == 0:
            predict_correct_covid += 1
        elif y_pred_RF[i] == y_test[i] and y_pred_RF[i] == 1:
            predict_correct_noncovid += 1
        elif y_pred_RF[i] != y_test[i] and y_pred_RF[i] == 0:
            predict_wrong_covid += 1
        elif y_pred_RF[i] != y_test[i] and y_pred_RF[i] == 1:
            predict_wrong_noncovid += 1

    # print("predict_correct_covid", predict_correct_covid)
    # print("predict_wrong_covid", predict_wrong_covid)
    # print("predict_correct_noncovid", predict_correct_noncovid)
    # print("predict_wrong_noncovid", predict_wrong_noncovid)
    # print("percen of correct covid", predict_correct_covid/(predict_correct_covid + predict_wrong_covid))
    # print("percen of correct noncovid", predict_correct_noncovid/(predict_correct_noncovid + predict_wrong_noncovid))
    # print("precent over all", (predict_correct_covid + predict_correct_noncovid)/len_of_y)
    # print("---------------------------------------------------------------------------------------------------\n")
    # print(y_pred_prob_RF)

    acc[3][loop] = (predict_correct_covid + predict_correct_noncovid) / len_of_y
    tp[3][loop] = predict_correct_covid
    tn[3][loop] = predict_correct_noncovid
    fp[3][loop] = predict_wrong_covid
    fn[3][loop] = predict_wrong_noncovid
    avg_time[3][loop] = end - start

    ###################################################KNN####################################################
    start = time.time()

    neigh = KNeighborsClassifier(n_neighbors=15)
    neigh.fit(X_train, y_train)

    end = time.time()
    # print("train time", end - start)

    y_pred_KNN = neigh.predict(X_test)

    # print(y_pred)
    # print(y_test)
    len_of_y = len(y_pred_KNN)
    predict_correct_covid = 0
    predict_wrong_covid = 0
    predict_correct_noncovid = 0
    predict_wrong_noncovid = 0

    for i in range(len_of_y):
        if y_pred_KNN[i] == y_test[i] and y_pred_KNN[i] == 0:
            predict_correct_covid += 1
        elif y_pred_KNN[i] == y_test[i] and y_pred_KNN[i] == 1:
            predict_correct_noncovid += 1
        elif y_pred_KNN[i] != y_test[i] and y_pred_KNN[i] == 0:
            predict_wrong_covid += 1
        elif y_pred_KNN[i] != y_test[i] and y_pred_KNN[i] == 1:
            predict_wrong_noncovid += 1

    # print("predict_correct_covid", predict_correct_covid)
    # print("predict_wrong_covid", predict_wrong_covid)
    # print("predict_correct_noncovid", predict_correct_noncovid)
    # print("predict_wrong_noncovid", predict_wrong_noncovid)
    # print("percen of correct covid", predict_correct_covid/(predict_correct_covid + predict_wrong_covid))
    # print("percen of correct noncovid", predict_correct_noncovid/(predict_correct_noncovid + predict_wrong_noncovid))
    # print("precent over all", (predict_correct_covid + predict_correct_noncovid)/len_of_y)

    acc[4][loop] = (predict_correct_covid + predict_correct_noncovid) / len_of_y
    tp[4][loop] = predict_correct_covid
    tn[4][loop] = predict_correct_noncovid
    fp[4][loop] = predict_wrong_covid
    fn[4][loop] = predict_wrong_noncovid
    avg_time[4][loop] = end - start

print(acc)
print(tp)
print(tn)
print(fp)
print(fn)
print(avg_time)

print("acc of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(acc, axis=1))
print("tp of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(tp, axis=1))
print("tn of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(tn, axis=1))
print("fp of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(fp, axis=1))
print("fn of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(fn, axis=1))
print("avg time of SVM, Bayesian, DecisionTree, RandomForest, KNN", np.mean(avg_time, axis=1))

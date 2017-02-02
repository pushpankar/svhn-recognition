from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

def one_hot_encode(labels):
    b = np.zeros((labels.shape[0],10))
    b[np.arange(labels.shape[0]), labels[1]] = 1
    return b

def get_train_data(path):
    train = loadmat(path+"train_32x32.mat")
    Xtrain = np.rollaxis(train['X'],3)
    ytrain = one_hot_encode(train['y'])
    return Xtrain, ytrain

def get_test_data(path):
    test = loadmat(path+"test_32x32.mat")
    Xtest = np.rollaxis(test['X'].astype(np.float32),3)
    ytest  = one_hot_encode(test['y'])
    return Xtest, ytest

def train_and_validation_data(path):
    #train test split
    data, labels = get_train_data(path)
    return train_test_split(data, labels, test_size=0.025, random_state=8)
    



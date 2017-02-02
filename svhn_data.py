from scipy.io import loadmat
import h5py
import os.path
import numpy as np
import cPickle as pickle
from os import listdir
from PIL import Image
from sklearn.model_selection import train_test_split

f = h5py.File('train/digitStruct.mat')
metadata= {}
metadata['height'] = []
metadata['label'] = []
metadata['left'] = []
metadata['top'] = []
metadata['width'] = []

def print_attrs(name, obj):
    vals = []
    if obj.shape[0] == 1:
        vals.append(obj[0][0])
    else:
        for k in range(obj.shape[0]):
            vals.append(f[obj[k][0]][0][0])
    metadata[name].append(vals)

def prepare_data():
    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)

    pickle_file = 'train/train_metadata.pickle'
    try:
        with open(pickle_file, 'wb') as f:
            f.dump(metadata, pickle_file)
    except Exception as e:
        print 'Unable to save data to', pickle_file, ':', e
        raise


def one_hot_encode(labels):
    b = np.zeros((len(labels),6,11))
    b[:,:,10] = 1
    for img_num in range(len(labels)):
        for index, num in enumerate(labels[img_num]):
            b[img_num,index,num] = 1
            b[img_num,index,10] = 0
    return b

def pad_images(image):
    a = np.random.rand(128,256,2)
    left = (128 - image.shape[0])//2
    top = (256 - image.shape[1])//2
    a[left:left+image.shape[0],top:top+image.shape[1]] = image
    return a

def get_train_data(path, offset, batch_size):
    if not os.path.exists(path + 'train_metadata.pickle'):
        prepare_data()
    with open('train/train_metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)
    
    imagelist = filter(lambda x:'png' in x, listdir(path))
    imagelist = sorted(imagelist, key=lambda x: int(filter(str.isdigit, x)))[offset:]
    small_images_indices = []
    loaded_images = []
    index = offset
    while len(small_images_indices) < batch_size:
        im = np.asarray(Image.open(path+imagelist[index]).convert('LA'))
        index += 1
        if im.shape[0] < 128 and im.shape[1] < 256:
            a = pad_images(im)
            print(a.shape)
            loaded_images.append(a)
            small_images_indices.append(index)

    ytrain = np.array(metadata['label'])[small_images_indices]
    ytrain = one_hot_encode(ytrain)
    Image.open(path+imagelist[small_images_indices[0]]).show()
    print(ytrain[0])
    return np.array(loaded_images), np.array(ytrain)

def get_test_data(path):
    test = loadmat(path+"test_32x32.mat")
    Xtest = np.rollaxis(test['X'].astype(np.float32),3)
    ytest  = one_hot_encode(test['y'])
    return Xtest, ytest

def train_and_validation_data(path):
    #train test split
    data, labels = get_train_data(path,)
    return train_test_split(data, labels, test_size=0.025, random_state=8)


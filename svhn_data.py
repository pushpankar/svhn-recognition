import h5py
import os.path
import numpy as np
import cPickle as pickle
from os import listdir
from PIL import Image

f = h5py.File('train/digitStruct.mat')
metadata = {}
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


def prepare_data(path):
    f = h5py.File('train/digitStruct.mat')
    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)

    pickle_file = path+'metadata.pickle'
    try:
        with open(pickle_file, 'wb') as pf:
            pickle.dump(metadata, pf)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def one_hot_encode(labels):
    b = np.zeros((len(labels), 6, 11))
    b[:, :, 10] = 1
    for img_num in range(len(labels)):
        for index, num in enumerate(labels[img_num]):
            b[img_num, index, num] = 1
            b[img_num, index, 10] = 0
    return b


def get_bounding_box_as_array(metadata, offset, batch_size):
    metadata['top'] = metadata['top'][offset:offset+batch_size]
    metadata['left'] = metadata['left'][offset:offset+batch_size]
    metadata['height'] = metadata['height'][offset:offset+batch_size]
    metadata['width'] = metadata['width'][offset:offset+batch_size]
    bbox = np.zeros((batch_size, 6, 4))

    for key in metadata:
        if key != 'label':
            for img_num in range(batch_size):
                for index, val in enumerate(metadata[key][img_num]):
                    if key == 'top':
                        label_num = 0
                    elif key == 'left':
                        label_num = 1
                    elif key == 'height':
                        label_num = 2
                    elif key == 'width':
                        label_num = 3
                    bbox[img_num, index, label_num] = val
    return bbox


def get_train_data(path, offset, batch_size):
    if not os.path.exists(path + 'metadata.pickle'):
        prepare_data(path)
    with open(path + 'metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)

    if not os.path.isfile(path+'imagelist.pickle'):
        imagelist = filter(lambda x: 'png' in x, listdir(path))
        imagelist = sorted(imagelist,
                           key=lambda x: int(filter(str.isdigit, x)))
        with open(path+'imagelist.pickle', 'wb') as f:
            pickle.dump(imagelist, f)
    else:
        with open(path+'imagelist.pickle', 'rb') as f:
            imagelist = pickle.load(f)

    loaded_images = []
    for image in imagelist[offset:offset+batch_size]:
        img = Image.open(path+image).convert('L').resize(
            (128, 32), Image.BILINEAR)
        im = np.asarray(img)
        loaded_images.append(im.reshape(128, 32, 1))

    ytrain = metadata['label'][offset:offset+batch_size]
    ytrain = one_hot_encode(ytrain)
    bbox = get_bounding_box_as_array(metadata, offset, batch_size)
    return np.array(loaded_images), np.array(ytrain), bbox

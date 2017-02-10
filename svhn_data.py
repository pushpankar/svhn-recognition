import tarfile
import h5py
import os.path
import numpy as np
import cPickle as pickle
from os import listdir
from PIL import Image
from six.moves.urllib.request import urlretrieve

url = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
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


def maybe_download(filename, path, expected_bytes):
    """Download file if not present in the current directory"""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
        with tarfile.open(filename) as f:
            f.extractall(path=path)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename)
    return filename


def one_hot_encode(labels):
    b = np.zeros((len(labels), 6, 11))
    b[:, :, 10] = 1
    for img_num in np.arange(len(labels)):
        for index, num in enumerate(labels[img_num]):
            b[img_num, index, num] = 1
            b[img_num, index, 10] = 0
    return b


def pad_list(l):
    y = np.array([x + [0] * (6 - len(x)) for x in l])
    return y


def get_bounding_box_as_array(metadata, offset, batch_size):
    for key in metadata:
        metadata[key] = pad_list(metadata[key][offset:offset+batch_size])
    bbox = np.zeros((batch_size, 6, 4))

    bbox[:, :, 0] = metadata['top']
    bbox[:, :, 1] = metadata['left']
    bbox[:, :, 2] = metadata['height']
    bbox[:, :, 3] = metadata['width']
    return bbox


def get_train_data(path, offset, batch_size):
    if not os.path.exists(path + 'metadata.pickle'):
        maybe_download("train.tar.gz", path, 404141560)
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

    # Image.open(path+imagelist[0]).show()
    loaded_images = []
    for image in imagelist[offset:offset+batch_size]:
        with Image.open(path+image) as img:
            img = img.convert('L').resize((128, 32), Image.BILINEAR)
            im = np.asarray(img)
            loaded_images.append(im.reshape(128, 32, 1))

    ytrain = metadata['label'][offset:offset+batch_size]
    ytrain = one_hot_encode(ytrain)
    bbox = get_bounding_box_as_array(metadata, offset, batch_size)
    return np.array(loaded_images), np.array(ytrain), bbox


def get_camera_images():
    imagelist = listdir('camera-pic/')
    loaded_images = []
    for image in imagelist:
        with Image.open('camera-pic/'+image) as img:
            img = img.convert('L').resize((128, 32), Image.BILINEAR)
            im = np.asarray(img)
            loaded_images.append(im.reshape(128, 32, 1))
    return np.array(loaded_images)

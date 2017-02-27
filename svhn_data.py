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
image_width = 32
image_height = 32


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
    b = np.zeros((len(labels), 5, 11))
    b[:, :, 10] = 1
    for img_num in np.arange(len(labels)):
        for index, num in enumerate(labels[img_num]):
            b[img_num, index, num] = 1
            b[img_num, index, 10] = 0
    return b


def pad_list(l):
    while len(l) < 5:
        l = np.append(l, [0])
    return np.array(l)


def bbox_as_array(bbox):
    new_bbox = np.zeros((5, 4))
    for i, point in enumerate(bbox):
        new_bbox[:, i] = pad_list(point)
    return new_bbox


def valid_ratio(img):
    wt, ht = img.size
    ratio = wt / float(ht)
    if ratio > 0.6 and ratio < 1.7:
        return True
    return False


def img_to_array(img):
    img = img.convert('L').resize((image_height, image_width), Image.BICUBIC)
    im = np.asarray(img)
    return im.reshape(image_height, image_width, 1)


def get_data(path, offset, batch_size):
    # Confirm that labels has been preprocessed
    if not os.path.exists(path + 'metadata.pickle'):
        maybe_download("train.tar.gz", path, 404141560)
        prepare_data(path)
    with open(path + 'metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)

    # create a list of images and sort them
    if not os.path.isfile(path+'imagelist.pickle'):
        imagelist = filter(lambda x: 'png' in x, listdir(path))
        imagelist = sorted(imagelist,
                           key=lambda x: int(filter(str.isdigit, x)))
        with open(path+'imagelist.pickle', 'wb') as f:
            pickle.dump(imagelist, f)
    else:
        with open(path+'imagelist.pickle', 'rb') as f:
            imagelist = pickle.load(f)

    # Get batch_size//2 images from the train set
    # And remaining as augmented long numbers
    loaded_images = []
    ytrain = []
    bbox = []
    imagelist = imagelist[offset:]
    label_list = metadata['label'][offset:]
    i = -1
    while len(loaded_images) < batch_size:
        i += 1
        image = imagelist[i]
        label = label_list[i]
        if (len(label) >= 3 and len(label) < 6):
            with Image.open(path+image) as img:
                img, bounds = crop_images(img, metadata, i+offset)
                im = img_to_array(img)
                loaded_images.append(im)
                ytrain.append(label)
                bbox.append(bbox_as_array(bounds))

    ytrain = one_hot_encode(ytrain)
    loaded_images = np.array(loaded_images)
    bbox = np.array(bbox)
    return loaded_images, ytrain, bbox


def crop_images(img, metadata, i):
    height = metadata['height'][i]
    width = metadata['width'][i]
    top = metadata['top'][i]
    left = metadata['left'][i]
    min_left = min(left)
    min_top = min(top)
    max_right = max(left) + width[np.argmax(left)]
    max_bottom = max(top) + height[np.argmax(top)]
    img = img.crop((min_left, min_top, max_right, max_bottom))
    return img, (np.array(top) - min_top, np.array(left) - min_left,
                 height, width)


def get_camera_images():
    imagelist = listdir('camera-pic/')
    loaded_images = []
    for image in imagelist:
        with Image.open('camera-pic/'+image) as img:
            img = img.convert('L').resize((image_height, image_width),
                                          Image.BILINEAR)
            im = np.asarray(img)/255.0
            loaded_images.append(im.reshape(image_height, image_width, 1))
    return np.array(loaded_images)

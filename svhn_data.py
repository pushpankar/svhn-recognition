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
image_width = 160
image_height = 160


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
    y = np.array([x + [0] * (5 - len(x)) for x in l])
    return y


def get_bounding_box_as_array(metadata, offset, batch_size):
    for key in metadata:
        metadata[key] = pad_list(metadata[key][offset:offset+batch_size])
    bbox = np.zeros((batch_size, 5, 4))

    bbox[:, :, 0] = metadata['top']
    bbox[:, :, 1] = metadata['left']
    bbox[:, :, 2] = metadata['height']
    bbox[:, :, 3] = metadata['width']
    return bbox


def valid_ratio(img):
    wt, ht = img.size
    ratio = wt / float(ht)
    if ratio > 0.6 and ratio < 1.7:
        return True
    return False


def img_to_array(img):
    img = img.convert('L').resize((image_height, image_width), Image.BICUBIC)
    im = np.asarray(img)
    return im.reshape(image_height, image_width)


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
    imagelist = imagelist[offset:]
    label_list = metadata['label'][offset:]
    i = 0
    while len(loaded_images) < batch_size:
        image = imagelist[i]
        label = label_list[i]
    # for image in imagelist[offset:offset+batch_size]:
        with Image.open(path+image) as img:
            i += 1
            if valid_ratio(img):
                im = img_to_array(img)
                loaded_images.append(im)
                ytrain.append(label)

    long_images, long_labels = get_long_numbers(path, imagelist, metadata,
                                                batch_size)
    ytrain = np.concatenate((np.array(long_labels), np.array(ytrain)), axis=0)
    loaded_images = np.concatenate((np.array(loaded_images), long_images),
                                   axis=0)
    ytrain = one_hot_encode(ytrain)
    # @TODO: Fix bounding boxes
    bbox = get_bounding_box_as_array(metadata, offset, batch_size)
    print(loaded_images.shape, ytrain.shape)
    return loaded_images, ytrain, bbox


def augment_dataset(images, labels, bbox, long_num_index):
    for i in long_num_index:
        for _ in range(4):
            angle = np.random.randint(15)
            image = Image.fromarray(images[i].reshape(image_height, image_width))
            images.append(np.asarray(image.rotate(angle)).reshape(image_height,
                                                                  image_width,
                                                                  1))
            images.append(np.asarray(image.rotate(-angle)).reshape(image_height,
                                                                   image_height,
                                                                   1))
            for _ in range(2):
                labels = np.append(labels, [labels[i]], axis=0)
                bbox = np.append(bbox, [bbox[i]], axis=0)
    return images, labels, bbox


def get_long_numbers(path, imagelist, metadata, size):
    labels = metadata['label']
    # filter long numbers
    long_num_index = np.argsort(map(len, labels))[-100:]
    long_num_index = np.random.choice(long_num_index, size=size)
    loaded_images = []
    image_labels = []
    for i in long_num_index:
        with Image.open(path + imagelist[i]) as img:
            loaded_images.append(img_to_array(img))
            image_labels.append(labels[i])
    return np.array(loaded_images), np.array(image_labels)


def get_camera_images():
    imagelist = listdir('camera-pic/')
    loaded_images = []
    for image in imagelist:
        with Image.open('camera-pic/'+image) as img:
            img = img.convert('L').resize((image_height, image_width),
                                          Image.BILINEAR)
            im = np.asarray(img)/255.0
            loaded_images.append(im.reshape(image_height, image_width))
    return np.array(loaded_images)

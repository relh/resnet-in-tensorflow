# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tarfile
from six.moves import urllib
import sys
import numpy as np
import cPickle
import os
import cv2
from scipy import misc
import PIL 

data_dir = 'chestxray8_data/'
image_dir = data_dir + 'images/'
label_dir = data_dir + 'labels/'
label_path = label_dir + 'Data_Entry_2017.csv'

#data_dir = 'cifar10_data'
#full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
#vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
#DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Todo rewrite
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_DEPTH = 1
NUM_CLASS = 14

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

BATCH_SIZE = 250 # How many batches of files you want to read in, from 0 to 5)
NUM_TRAIN_BATCH = 2 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH


def _read_one_batch(path, is_random_label, batch_size=BATCH_SIZE):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    data = np.zeros((BATCH_SIZE, 1024*1024*1))
    problems = ['No Finding', 'Pneumothorax', 'Effusion', 'Cardiomegaly', 'Pleural_Thickening', 'Atelectasis', 'Consolidation', 'Edema', 'Emphysema', 'Pneumonia', 'Nodule', 'Mass', 'Infiltration', 'Hernia', 'Fibrosis']
    label = np.zeros((BATCH_SIZE, len(problems)-1))
    encoding = np.eye(len(problems)-1)

    line_count = -1
    for line in open(path,'r'):
        #Image Index    Finding Labels  Follow-up # Patient ID  Patient Age Patient Gender  View Position   OriginalImage[Width Height] OriginalImagePixelSpacing[x y]
        line_count += 1
        if line_count == 0:
            continue 
        if line_count == BATCH_SIZE:
            break

        name, labels, followup, id, age, gender, view, orig_width, orig_height, orig_space_x, orig_space_y, = line.rstrip().split(',')
        image_path = image_dir + name 

        print labels
        # Only load in images that exist
        if os.path.exists(image_path):
            image = misc.imread(image_path)
            if len(image.shape) > 2:
                #TODO, what is this strange shape problem? Revisit tonight.
                print name + " has broken dimensions!"
                continue
            data[line_count] = image.flatten()

            diagnoses = labels.split('|')
            for problem in diagnoses:
                if problem == 'No Finding':
                    continue 
                label[line_count] += encoding[problems.index(problem)-1]

    print data[1]
    print label[1]
    return data, label


def read_in_all_images(address_list=[label_path], shuffle=True, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([]).reshape([0, NUM_CLASS])

    for address in address_list:
        print 'Reading images from ' + address
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print 'Shuffling'
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(label_path) #full_data_dir + str(i))
    data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    
    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels

if __name__ == "__main__":
    read_in_all_images()
    #_read_one_batch(label_path, False)

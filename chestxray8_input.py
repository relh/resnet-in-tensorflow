import tarfile
from six.moves import urllib
import sys
import pickle
import numpy as np
import os
import cv2
from scipy import misc
import PIL 

data_dir = 'chestxray8_data/'
image_dir = data_dir + 'images/'
label_dir = data_dir + 'labels/'
label_path = label_dir + 'Data_Entry_2017.csv'

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_DEPTH = 1
NUM_CLASS = 14

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

BATCH_SIZE = 3 # How many batches of files you want to read in, from 0 to 5)
# Total number of images is 50,000
TRAIN_SIZE = 50000

problems = ['No Finding', 'Pneumothorax', 'Effusion', 'Cardiomegaly', 'Pleural_Thickening', 'Atelectasis', 'Consolidation', 'Edema', 'Emphysema', 'Pneumonia', 'Nodule', 'Mass', 'Infiltration', 'Hernia', 'Fibrosis']
encoding = np.eye(len(problems)-1)#, dtype=int)

def load_images(idx_range, image_labels, shuffle=True):
    #def load_in_all_images(address_list=[label_path], shuffle=True, is_random_label = False):
    #images = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    #labels = np.array([]).reshape([0, NUM_CLASS])
    images = np.zeros((len(idx_range), IMG_WIDTH*IMG_HEIGHT*IMG_DEPTH))
    labels = np.zeros((len(idx_range), NUM_CLASS))

    for batch_idx, idx in enumerate(idx_range):
      #name, labels, followup, id, age, gender, view, orig_width, orig_height, orig_space_x, orig_space_y, = line.rstrip().split(',')
      path = image_labels[idx][0]
      label = image_labels[idx][1]
      image_path = image_dir + path 

      print("Loading {}/{}.. {}".format(idx, len(idx_range), label))
      # Only load in images that exist
      if os.path.exists(image_path):
          image = misc.imread(image_path)
          if len(image.shape) > 2:
              print(name + " has broken dimensions!")
              continue
          images[batch_idx] = image.flatten()

          diagnoses = label.split('|')
          for problem in diagnoses:
              if not problem == 'No Finding':
                labels[batch_idx] += encoding[problems.index(problem)-1]
      else:
          print(image_path + " does not exist!")

    # Get images here
    #data = np.concatenate((data, batch_data))
    #label = np.concatenate((label, batch_label))
    num_data = len(labels)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    images = images.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    images = images.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        print('Shuffling')
        order = np.random.permutation(num_data)
        images = images[order, ...]
        labels = labels[order]

    images = images.astype(np.float32)

    print(images.shape)
    print(labels.shape)
    return images, labels

def prepare_train_data(batch_size=BATCH_SIZE, padding_size=0, path=label_path, shuffle=True, is_random_label=False):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels

    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]

    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''

    if os.path.exists('image_labels.p'):
      image_labels = pickle.load(open('image_labels.p', 'rb'))
    else:
      image_labels = []
      skip = -1
      read_count = 0
      for line in open(path,'r'):
          #Image Index    Finding Labels  Follow-up # Patient ID  Patient Age Patient Gender  View Position   OriginalImage[Width Height] OriginalImagePixelSpacing[x y]
          skip += 1
          if skip == 0:
              continue 
          if read_count == TRAIN_SIZE:
              break

          name, labels, followup, id, age, gender, view, orig_width, orig_height, orig_space_x, orig_space_y, = line.rstrip().split(',')
          image_path = image_dir + name 

          print("{}/{}.. {}".format(read_count, TRAIN_SIZE, labels))
          # Only load in images that exist
          if os.path.exists(image_path):
              image = misc.imread(image_path)
              if len(image.shape) > 2:
                  print(name + " has broken dimensions!")
                  continue
              image_labels.append((name, labels))
              read_count += 1
          else:
              print(image_path + " does not exist!")
      pickle.dump(image_labels, open('image_labels.p', 'wb'))

    return image_labels 


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

if __name__ == "__main__":
    image_labels = prepare_train_data()
    load_images(range(1,10), image_labels)

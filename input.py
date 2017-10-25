import tarfile
from six.moves import urllib
import sys
import csv
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
TRAIN_SIZE = 40000
VAL_SIZE = 10000

problems = ['No Finding', 'Pneumothorax', 'Effusion', 'Cardiomegaly', 'Pleural_Thickening', 'Atelectasis', 'Consolidation', 'Edema', 'Emphysema', 'Pneumonia', 'Nodule', 'Mass', 'Infiltration', 'Hernia', 'Fibrosis']
encoding = np.eye(len(problems)-1)#, dtype=int)

def load_images(idx_range, image_labels, shuffle=True):
    images = np.zeros((len(idx_range), IMG_WIDTH*IMG_HEIGHT*IMG_DEPTH))
    labels = np.zeros((len(idx_range), NUM_CLASS))

    for batch_idx, idx in enumerate(idx_range):
      path = image_labels[idx][0]
      label = image_labels[idx][1]
      image_path = image_dir + path 

      #print("Loading {}/{}.. {}".format(idx, len(idx_range), label))
      # Only load in images that exist
      if os.path.exists(image_path):
          image = misc.imread(image_path)
          if len(image.shape) > 2:
              print(name + " has broken dimensions!")
              continue
          try:
              images[batch_idx] = image.flatten()
          except:
              print('Excepted!')
              continue

          diagnoses = label.split('|')
          for problem in diagnoses:
              if not problem == 'No Finding':
                labels[batch_idx] += encoding[problems.index(problem)-1]
      else:
          print(image_path + " does not exist!")

    # Get images here
    num_data = len(labels)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    images = images.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    images = images.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        #print('Shuffling')
        order = np.random.permutation(num_data)
        images = images[order, ...]
        labels = labels[order]

    images = images.astype(np.float32)

    #print(images.shape)
    #print(labels.shape)
    return images, labels


def prepare_data(mode, prep_size, padding_size=0, path=label_path):
    if os.path.exists(mode+'_paths.p'):
      image_labels = pickle.load(open(mode+'_paths.p', 'rb'))
    else:
      image_labels = []
      read_count = 0
      with open(path, 'r') as textfile:
          # Go backwards for validation to avoid overlap
          if mode == 'val':
              data = reversed(list(csv.reader(textfile)))
          else:
              data = list(csv.reader(textfile))
              data.pop(0) # remove headers

      for line in data: #open(path,'r'):
          #Image Index    Finding Labels  Follow-up # Patient ID  Patient Age Patient Gender  View Position   OriginalImage[Width Height] OriginalImagePixelSpacing[x y]
          if read_count == prep_size:
              break

          name, labels, followup, id, age, gender, view, orig_width, orig_height, orig_space_x, orig_space_y, = line
          image_path = image_dir + name 

          print("{}/{}.. {}".format(read_count, prep_size, labels))
          # Only load in images that exist
          if os.path.exists(image_path):
              image = misc.imread(image_path)
              if len(image.shape) > 2:
                  print(name + " has broken dimensions!")
                  continue
              try:
                image.flatten()
              except TypeError as e:
                continue
          
              image_labels.append((name, labels))
              read_count += 1
          else:
              print(image_path + " does not exist!")
      pickle.dump(image_labels, open(mode+'_paths.p', 'wb'))
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

def get_random_indices(prep_size, batch_size=3):
    offset = np.random.choice(prep_size - batch_size, 1)[0]
    #batch_data = train_data[offset:offset+train_batch_size, ...]
    #batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
    #batch_data = whitening_image(batch_data)
    #batch_label = train_labels[offset:offset+train_batch_size]

    indices = range(offset,offset+batch_size)
    return indices

if __name__ == "__main__":
    image_labels = prepare_data('val', VAL_SIZE)
    image_labels = prepare_data('train', TRAIN_SIZE)
    load_images(range(1,10), image_labels)

import numpy as np
from scipy import misc
import PIL 
import os

data_dir = 'chestxray8_data/'
image_dir = data_dir + 'images/'
label_dir = data_dir + 'labels/'
label_path = label_dir + 'Data_Entry_2017.csv'

skip_line = 0
for line in open(label_path,'r'):
    #Image Index    Finding Labels  Follow-up # Patient ID  Patient Age Patient Gender  View Position   OriginalImage[Width Height] OriginalImagePixelSpacing[x y]
    if skip_line is 0:
        skip_line = 1
        continue 

    name, labels, followup, id, age, gender, view, orig_width, orig_height, orig_space_x, orig_space_y, = line.rstrip().split(',')
    image_path = image_dir + name 

    # Only load in images that exist
    if os.path.exists(image_path):
        image = misc.imread(image_path)
        print image.shape

        # Avoid images with weird shapes
        if len(image.shape) > 2:
            print name

    # TODO, what is this strange shape problem? Revisit tonight.

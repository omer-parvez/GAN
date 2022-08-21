# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from glob import glob
# loading the directories
training_dir = r'D:\OneDrive - Universitetet i Stavanger\Masters Thesis\output\train'
validation_dir = r'D:\OneDrive - Universitetet i Stavanger\Masters Thesis\output\val'
test_dir = r'D:\OneDrive - Universitetet i Stavanger\Masters Thesis\output\test'
# useful for getting number of files
image_files = glob(training_dir + '/*/*.jp*g')
valid_image_files = glob(validation_dir + '/*/*.jp*g')
image = cv2.imread(image_files[0])
print(image.shape)
print (len (valid_image_files))
# getting the number of classes i.e. type of fruits
folders = glob(training_dir + '/*')
num_classes = len(folders)
print ('Total Classes = ' + str(num_classes))
# importing the libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
#from keras.preprocessing import image

IMAGE_SIZE = [128, 128]  # we will keep the image size as (64,64). You can increase the size for better results.

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
#x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Image Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
print (type (training_generator[0].shear))
# The labels are stored in class_indices in dictionary form.
# checking the labels
training_generator.class_indices
# training_images = 37836
# validation_images = 12709
batch_size = 200
history = model.fit(training_generator,
                   steps_per_epoch = len(image_files)//batch_size,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results.
                   epochs = 100,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = len(valid_image_files)//batch_size)  # this should be equal to total number of images in validation set.
print ('Training Accuracy = ' + str(history.history['acc']))
print ('Validation Accuracy = ' + str(history.history['val_acc']))
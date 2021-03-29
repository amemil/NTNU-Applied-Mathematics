#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 18:56:30 2021

@author: emilam
"""

## WEEK 1 ##
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
'''
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer = 'sgd',loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs,ys,epochs = 500)
print(model.predict([10.0]))

# GRADED FUNCTION: house_model

#model: x-room apartment has price: y = 0.5 + 0.5x (in hundred thousands)
def house_model(y_new):
    xs = np.array([1,2,3,4,5,6])
    ys = np.array([1,1.5,2,2.5,3,3.5])
    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer = 'sgd',loss='mean_squared_error')
    model.fit(xs,ys,epochs=2000)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)
'''
## WEEK 2 ##
'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()


#28x28 input (pictures), 10 classes
model2 = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),\
                          keras.layers.Dense(128,activation=tf.nn.relu),\
                              keras.layers.Dense(10,activation=tf.nn.softmax)])
    
#Explore pictures and data with matplotlib, pixels values 0-255    
plt.imshow(train_images[10]) 
print(train_labels[10])
print(train_images[10])

#Normalise data

train_images = train_images / 255.0
test_images = test_images / 255.0

#Fit and train model

model2.compile(optimizer = tf.train.AdamOptimizer(),loss = 'sparse_categorical_crossentropy')
model2.fit(train_images,train_labels,epochs = 10)

#Try on test data

model2.evaluate(test_images,test_labels)

#Callbacks to stop training at some threshold:
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.9):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True
            
callbacks = myCallback()
model2.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model2.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])
'''

#MNIST DATASET
'''
def train_mnist():
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.90):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()


    mnist = keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    plt.imshow(x_train[1]) 
    
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),\
                          keras.layers.Dense(128,activation=tf.nn.relu),\
                              keras.layers.Dense(10,activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    # model fitting
    history = model.fit(x_train,y_train,epochs=10, callbacks=[callbacks])
    # model fitting
    print(model.predict(x_train)[1])
    return history.epoch, history.history['acc'][-1]

train_mnist()
'''

# WEEEK 3 - CONVOLUTIONAL LAYERS/NETWORKS
'''
model4 = keras.models.Sequential([keras.layers.Conv2D(64, (3,3), activation= 'relu',input_shape=(28,28,1)),\
                                  keras.layers.MaxPooling2D(2,2),\
                                      keras.layers.Conv2D(64, (3,3), activation= 'relu'),\
                                          keras.layers.MaxPooling2D(2,2),\
                                              keras.layers.Flatten(),\
                                                  keras.layers.Dense(128,activation='relu'),\
                                                      keras.layers.Dense(10,activation='softmax')])

model4.summary()


#Back to Fashion MNIST data from week 1, improved convolutional model

mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = keras.models.Sequential([
  keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=20)
test_loss = model.evaluate(test_images, test_labels)

#Visualization of the convolutions

f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
  
  
  
#PLAY MORE WITH FILTERS,CONVOLUTIONS AND POOLING

import cv2
from scipy import misc
i = misc.ascent() #random picture from scipy to work with

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

#Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# CREATE CONVOLUTION
for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x - 1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[0][1])
      convolution = convolution + (i[x + 1, y-1] * filter[0][2])
      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution

#PLOT AFTER APPLYING FILTER/CONV      
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()   


## POOLING: REDUCE SIZE OF PICTURE
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()      
'''

#EXERCISE 3: 1 CONVOLUTIONAL LAYER TO MAKE 99.8% ACCURACY ON TRAINING ON MNIST DATA
'''
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()

    mnist = keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0

    model = keras.models.Sequential([keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
      keras.layers.MaxPooling2D(2, 2),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images,training_labels,epochs=20, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]

_, _ = train_mnist_conv()

'''

# WEEK 4: MORE COMPLEX IMAGES
    
#IMAGE GENERATOR, make directories with train and validation data and subdirectories with labels

import tensorflow.keras.preprocessing.image
import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,\
                                                    target_size=(300,300),\
                                                        batch_size=128,\
                                                            class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(validation_dir,\
                                                    target_size=(300,300),\
                                                        batch_size=128,\
                                                            class_mode='binary')

#do the same for validation/test set


#Model for more complex images, binary classification human / horse

model5 = keras.models.Sequential([keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),\
                                  keras.layers.MaxPooling2D(2,2),\
                                      keras.layers.Conv2D(32,(3,3),activation='relu'),\
                                          keras.layers.MaxPooling2D(2,2),\
                                              keras.layers.Conv2D(64,(3,3),activation='relu'),\
                                                  keras.layers.MaxPooling2D(2,2),\
                                                      keras.layers.Flatten(),\
                                                          keras.layers.Dense(512,activation='relu'),\
                                                              keras.layers.Dense(1,activation='sigmoid')])
    
from tensorflow.keras.optimizers import RMSprop

model5.compile(loss='binary_crossentropy',\
              optimizer=RMSprop(lr=0.001),\
                  metrics=['acc'])

history = model5.fit_generator(train_generator,steps_per_epoch = 8,epochs=15,\
                              validatoin_data = validation_generator,validation_Steps=8,vebose=2)
    
## WORKING WITH THE DATASET

#download train and test set

#!wget --no-check-certificate \
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
#    -O /tmp/horse-or-human.zip
   

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip
    
import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])


#display images:

import matplotlib.image as mpimg
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0    
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#Building the model

model6 = keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    # he third convolution
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    keras.layers.Flatten(),
    # 512 neuron hidden layer
    keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    keras.layers.Dense(1, activation='sigmoid')
])

model6.summary()

model6.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

#train model
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

#Running the model

from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
    
#Visualize convolutions

import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model6.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model6.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model6.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')



#EXERCISE 4: LEARN HAPPY OR SAD DATASET:
    
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()



def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.999):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model7 = keras.models.Sequential([keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),\
                                  keras.layers.MaxPooling2D(2,2),\
                                      keras.layers.Conv2D(32,(3,3),activation='relu'),\
                                          keras.layers.MaxPooling2D(2,2),\
                                              keras.layers.Conv2D(64,(3,3),activation='relu'),\
                                                  keras.layers.MaxPooling2D(2,2),\
                                                      keras.layers.Flatten(),\
                                                          keras.layers.Dense(512,activation='relu'),\
                                                              keras.layers.Dense(1,activation='sigmoid')])

    from tensorflow.keras.optimizers import RMSprop

    model7.compile(loss='binary_crossentropy',\
              optimizer=RMSprop(lr=0.001),\
                  metrics=['acc'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s/',
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')
    # Expected output: 'Found 80 images belonging to 2 classes'

    history = model7.fit_generator(train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,callbacks=[callbacks])
    return history.history['acc'][-1]


train_happy_sad_model()
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:30:43 2019

@author: lxi
"""

#%%

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

#%%

# you need to change this to your data directory
train_dir = 'C:\\xilong\\cat-dog\\01 cats vs dogs\\data_test\\train\\'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    planes = []
    label_planes = []
    
    # we define cats are belong to the category of 0, dogs 1 and plane 2.
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        if name[0]=='dog':
            dogs.append(file_dir + file)
            label_dogs.append(1)
        if name[0]=='plane':
            planes.append(file_dir + file)
            label_planes.append(2)
    #print(cats)
    #print(dogs)
    #print(planes)
    #print(label_planes)
    print('There are %d cats\nThere are %d dogs\nThere are %d planes' %(len(cats), len(dogs), len(planes)))
    
    image_list = np.hstack((cats, dogs, planes))
    label_list = np.hstack((label_cats, label_dogs, label_planes))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    
    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)  # Data type conversion, converting list type into tensor that tensorflow can understand
    label = tf.cast(label, tf.int32)   # Data type conversion

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use 


images, labels = get_files(train_dir)
print(images, labels)


image = tf.cast(images, tf.string)  # Data type conversion, converting list type into tensor that tensorflow can understand
label = tf.cast(labels, tf.int32)   # Data type conversion
 
input_queue = tf.train.slice_input_producer([image, label],num_epochs=None,shuffle=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    value = sess.run(input_queue)
    print(value)
    label = value[1]
    print(label)
    image_contents = tf.read_file(value[0])
    #print(image_contents.eval())
    
    # the image would convert into three dimensional matrix, which is a tensor actually.
    image = tf.image.decode_jpeg(image_contents, channels=3)
    print(image.eval())
    
    # show that picture decoded into tensor type
    plt.imshow(image.eval())
    plt.show()

    
#label = input_queue[1]
#print(label)
#
#image_contents = tf.read_file(input_queue[0])


#train_batch, train_label_batch = get_batch(images,labels,IMG_W,IMG_H,BATCH_SIZE, CAPACITY)
#with tf.Session() as sess:
#    print(sess.run([train_batch,train_label_batch])）
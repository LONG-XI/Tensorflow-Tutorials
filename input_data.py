#%%

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

#%%

# you need to change this to your data directory
train_dir = '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    number1 = []
    label_number1 = []
    number2 = []
    label_number2 = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')  # name : ['1', '12453', 'jpg'].  name[0]: 12121212121212111122222
        #print(name)
        #print(name[0])
        if name[0]=='1':
            number1.append(file_dir + file)
            label_number1.append(0)
        else:
            number2.append(file_dir + file)
            label_number2.append(1)
    #print(number1)  
    #['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.6633.jpg', '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.9500.jpg', ...,]
    #print(label_number1)  #[0,0,0,0,0,0,0......,0]
    print('There are %d number 1\nThere are %d number 2' %(len(number1), len(number2)))
    
    # np.hstack: combine two arrays, tiled horizontally
    image_list = np.hstack((number1, number2))
    label_list = np.hstack((label_number1, label_number2))
    
    temp = np.array([image_list, label_list])
    #print(temp)
    #[['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.6633.jpg'
    #  '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.9500.jpg'
    #  '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.4024.jpg'
    #  ...
    #  '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.1843.jpg'
    #  '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.12103.jpg'
    #  '/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.2392.jpg']
    #['0' '0' '0' ... '1' '1' '1']]
    
    
    temp = temp.transpose()
    
    
    #print(temp)
    #[['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.6633.jpg'
    #  '0']
    # ['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.9500.jpg'
    #  '0']
    # ['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/1.4024.jpg'
    #  '0']
    # ...
    # ['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.1843.jpg'
    #  '1']
    # ['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.12103.jpg'
    #  '1']
    # ['/Users/xilong/Desktop/gesture-recognization-tensorflow/data/train/2.2392.jpg'
    #  '1']]
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    #print(image_list)
    label_list = list(temp[:, 1])
    #print(label_list)
    label_list = [int(i) for i in label_list]  # change the type string to int
    
    
    return image_list, label_list   # all list type


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

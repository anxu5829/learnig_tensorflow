""" Solution for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = '.\\data\\mnist'
# utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data

# batch_size = 128 , 表示将tf.data.Dataset按照batch_size可以一块一块的取出来
# train_data 的类型变为：batchDataset
train_data = train_data.batch(batch_size)

# test_data 在预测时也是一块一块做的
# 因为这是dataset的特性
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)


# 使用Iterator拿到数据
# iterator 没有设定batch的信息
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)

img, label = iterator.get_next()

# 注意这是两个operator，当前并未执行
train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data


# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y



w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer


logits = tf.matmul(img, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')  # computes the mean over all the examples in the batch


# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)

# tf 定义了大量的操作
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:

    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    # 总之，tf的操作思路：数据是一点一点进来的
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()

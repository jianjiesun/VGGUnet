from layer import *
from mlxtend.data import loadlocal_mnist
import tensorflow as tf
import numpy as np
import cv2


inputs = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = 'input') 
ground_truth = tf.placeholder(tf.int32, shape = [None, 28, 28, 1], name = 'ground_truth')
keep_prob = tf.placeholder(tf.float32, name = 'dropout_probability')

conv_1 = conv_layer(inputs, conv_size = 3, output_depth = 64, name = 'layer1')
conv_1 = conv_layer(conv_1, conv_size = 3, output_depth = 64, name = 'layer2')
pool_1 = max_pooling_layer(conv_1, 2)
conv_2 = conv_layer(pool_1, conv_size = 3, output_depth = 128, name = 'layer3')
conv_2 = conv_layer(conv_2, conv_size = 3, output_depth = 128, name = 'layer4')
drop_2 = dropout_layer(conv_2, keep_prob)
pool_2 = max_pooling_layer(drop_2, 2)

conv_3 = conv_layer(pool_2, conv_size = 3, output_depth = 256, name = 'layer5')
conv_3 = conv_layer(conv_3, conv_size = 3, output_depth = 256, name = 'layer6')
drop_3 = dropout_layer(conv_3, keep_prob)

uppool_4 = deconv_layer(drop_3, conv_size = 2, output_depth = 128, stride = 2, name = 'layer7')
concat_4 = concat_layer(drop_2, uppool_4)
conv_4 = conv_layer(concat_4, conv_size = 3, output_depth = 128, name = 'layer8')
conv_4 = conv_layer(conv_4, conv_size = 3, output_depth = 128, name = 'layer9')

uppool_5 = deconv_layer(conv_4, conv_size = 2, output_depth = 64, stride = 2, name = 'layer10')
concat_5 = concat_layer(conv_1, uppool_5)
conv_5 = conv_layer(concat_5, conv_size = 3, output_depth = 64, name = 'layer11')
conv_5 = conv_layer(conv_5, conv_size = 3, output_depth = 64, name = 'layer12')
conv_6 = conv_layer(conv_5, conv_size = 1, output_depth = 1, name = 'layer13',
    activation_function = tf.nn.sigmoid, normalize = False)

loss = image_focal_loss(conv_6, ground_truth, 2, 0.25)

optimizer = tf.train.AdamOptimizer(1e-2)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
# Initialize variables
sess.run(init)


#load training data
image, label = loadlocal_mnist(
    images_path = ('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/tensorflow_test/'
        'mnist dataset/train-images.idx3-ubyte'), 
    labels_path = ('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/tensorflow_test/'
        'mnist dataset/train-labels.idx1-ubyte'))

#set batch size
batch_size = 100
batch_number = np.array(image.shape[0] / batch_size).astype(np.int32)



# #training
print('Training ------------')
program_end = False
for epoch in range(2):
    for batch in range(batch_number):
        batch_image = []
        for i in range(batch_size):
            single_image = image[batch * batch_size + i].reshape([28, 28, 1])
            threshold_index = single_image[:, :, :] > 10
            single_image[threshold_index] = 255
            single_image = single_image / 255

            try:
                batch_image.append(single_image)
            except:
                batch_image = single_image

                
        batch_image = np.array(batch_image)

        sess.run(train_step, feed_dict = {inputs: batch_image, ground_truth: batch_image, keep_prob: 0.5 })
        train_loss = sess.run(loss, feed_dict = {inputs: batch_image, ground_truth: batch_image,
            keep_prob: 1})

        test = sess.run(conv_6, feed_dict = {inputs: batch_image, keep_prob: 1})

        print('epoch:', epoch)
        print('batch:', batch)
        print('loss:', train_loss)

        if epoch == 1:
            test_single = (test[0] * 255).astype(np.uint8)
            print('image max value:',np.max(test_single))

            threshold_test_index = test_single[:,:,:] > 127
            threshold_test = np.zeros((28,28,1), dtype = np.uint8)
            threshold_test[threshold_test_index] = 255

            cv2.imshow('test', test_single)
            cv2.imshow('threshold', threshold_test)
            cv2.imshow('original', batch_image[0])
            if cv2.waitKey(0) == 27:
                program_end = True
                break
    if program_end:
        break
    
    # ckpt_name = "ckpt/cnn_model_" + str(epoch + 1) + ".ckpt"
    # save_path = saver.save(sess, ckpt_name)
    # print("model save path:", save_path)

import tensorflow as tf
import numpy as np
import cv2
import logging
from .layer import *


def Unet(inputs, keep_prob = 1):
    conv_1 = conv_layer(inputs, conv_size = 3, output_depth = 64, name = 'layer1')
    conv_1 = conv_layer(conv_1, conv_size = 3, output_depth = 64, name = 'layer2')
    pool_1 = max_pooling_layer(conv_1, 2)

    conv_2 = conv_layer(pool_1, conv_size = 3, output_depth = 128, name = 'layer3')
    conv_2 = conv_layer(conv_2, conv_size = 3, output_depth = 128, name = 'layer4')
    pool_2 = max_pooling_layer(conv_2, 2)

    conv_3 = conv_layer(pool_2, conv_size = 3, output_depth = 256, name = 'layer5')
    conv_3 = conv_layer(conv_3, conv_size = 3, output_depth = 256, name = 'layer6')
    pool_3 = max_pooling_layer(conv_3, 2)

    conv_4 = conv_layer(pool_3, conv_size = 3, output_depth = 512, name = 'layer7')
    conv_4 = conv_layer(conv_4, conv_size = 3, output_depth = 512, name = 'layer8')
    drop_4 = dropout_layer(conv_4, keep_prob)
    pool_4 = max_pooling_layer(drop_4, 2)

    conv_5 = conv_layer(pool_4, conv_size = 3, output_depth = 1024, name = 'layer9')
    conv_5 = conv_layer(conv_5, conv_size = 3, output_depth = 1024, name = 'layer10')
    drop_5 = dropout_layer(conv_5, keep_prob)

    deconv_6 = deconv_layer(drop_5, conv_size = 2, output_depth = 512, stride = 2, name = 'layer11')
    concat_6 = concat_layer(drop_4, deconv_6)
    conv_6 = conv_layer(concat_6, conv_size = 3, output_depth = 512, name = 'layer12')
    conv_6 = conv_layer(conv_6, conv_size = 3, output_depth = 512, name = 'layer13')

    deconv_7 = deconv_layer(conv_6, conv_size = 2, output_depth = 256, stride = 2, name = 'layer14')
    concat_7 = concat_layer(conv_3, deconv_7)
    conv_7 = conv_layer(concat_7, conv_size = 3, output_depth = 256, name = 'layer15')
    conv_7 = conv_layer(conv_7, conv_size = 3, output_depth = 256, name = 'layer16')

    deconv_8 = deconv_layer(conv_7, conv_size = 2, output_depth = 128, stride = 2, name = 'layer17')
    concat_8 = concat_layer(conv_2, deconv_8)
    conv_8 = conv_layer(concat_8, conv_size = 3, output_depth = 128, name = 'layer18')
    conv_8 = conv_layer(conv_8, conv_size = 3, output_depth = 128, name = 'layer19')

    deconv_9 = deconv_layer(deconv_8, conv_size = 2, output_depth = 64, stride = 2, name = 'layer20')
    concat_9 = concat_layer(conv_1, deconv_9)
    conv_9 = conv_layer(concat_9, conv_size = 3, output_depth = 64, name = 'layer21')
    conv_9 = conv_layer(conv_9, conv_size = 3, output_depth = 64, name = 'layer22')
    conv_10 = conv_layer(conv_9, conv_size = 1, output_depth = 1, name = 'layer23',
        activation_function = tf.nn.sigmoid, normalize = False)

    return conv_10


def VGGUnet(inputs, keep_prob = 1, parameter = None):
    conv_1 = conv_layer(inputs, conv_size = 3, output_depth = 64, name = 'layer1', parameter = parameter)
    conv_1 = conv_layer(conv_1, conv_size = 3, output_depth = 64, name = 'layer2', parameter = parameter)
    pool_1 = max_pooling_layer(conv_1, 2)

    conv_2 = conv_layer(pool_1, conv_size = 3, output_depth = 128, name = 'layer3', parameter = parameter)
    conv_2 = conv_layer(conv_2, conv_size = 3, output_depth = 128, name = 'layer4', parameter = parameter)
    pool_2 = max_pooling_layer(conv_2, 2)

    conv_3 = conv_layer(pool_2, conv_size = 3, output_depth = 256, name = 'layer5', parameter = parameter)
    conv_3 = conv_layer(conv_3, conv_size = 3, output_depth = 256, name = 'layer6', parameter = parameter)
    conv_3 = conv_layer(conv_3, conv_size = 3, output_depth = 256, name = 'layer7', parameter = parameter)
    pool_3 = max_pooling_layer(conv_3, 2)

    conv_4 = conv_layer(pool_3, conv_size = 3, output_depth = 512, name = 'layer8', parameter = parameter)
    conv_4 = conv_layer(conv_4, conv_size = 3, output_depth = 512, name = 'layer9', parameter = parameter)
    conv_4 = conv_layer(conv_4, conv_size = 3, output_depth = 512, name = 'layer10', parameter = parameter)
    drop_4 = dropout_layer(conv_4, keep_prob)
    pool_4 = max_pooling_layer(drop_4, 2)

    conv_5 = conv_layer(pool_4, conv_size = 3, output_depth = 512, name = 'layer11', parameter = parameter)
    conv_5 = conv_layer(conv_5, conv_size = 3, output_depth = 512, name = 'layer12', parameter = parameter)
    conv_5 = conv_layer(conv_5, conv_size = 3, output_depth = 512, name = 'layer13', parameter = parameter)
    drop_5 = dropout_layer(conv_5, keep_prob)

    deconv_6 = deconv_layer(drop_5, conv_size = 2, output_depth = 512, stride = 2, name = 'layer14')
    concat_6 = concat_layer(drop_4, deconv_6)
    conv_6 = conv_layer(concat_6, conv_size = 3, output_depth = 512, name = 'layer15')
    conv_6 = conv_layer(conv_6, conv_size = 3, output_depth = 512, name = 'layer16')

    deconv_7 = deconv_layer(conv_6, conv_size = 2, output_depth = 256, stride = 2, name = 'layer17')
    concat_7 = concat_layer(conv_3, deconv_7)
    conv_7 = conv_layer(concat_7, conv_size = 3, output_depth = 256, name = 'layer18')
    conv_7 = conv_layer(conv_7, conv_size = 3, output_depth = 256, name = 'layer19')

    deconv_8 = deconv_layer(conv_7, conv_size = 2, output_depth = 128, stride = 2, name = 'layer20')
    concat_8 = concat_layer(conv_2, deconv_8)
    conv_8 = conv_layer(concat_8, conv_size = 3, output_depth = 128, name = 'layer21')
    conv_8 = conv_layer(conv_8, conv_size = 3, output_depth = 128, name = 'layer22')

    deconv_9 = deconv_layer(deconv_8, conv_size = 2, output_depth = 64, stride = 2, name = 'layer23')
    concat_9 = concat_layer(conv_1, deconv_9)
    conv_9 = conv_layer(concat_9, conv_size = 3, output_depth = 64, name = 'layer24')
    conv_9 = conv_layer(conv_9, conv_size = 3, output_depth = 64, name = 'layer25')
    conv_10 = conv_layer(conv_9, conv_size = 1, output_depth = 1, name = 'layer26',
        activation_function = tf.nn.sigmoid, normalize = False)

    return conv_10


def fault_Unet(inputs, keep_prob = 1):
    conv_1 = conv_layer(inputs, conv_size = 3, output_depth = 64, name = 'layer1')
    conv_1 = conv_layer(conv_1, conv_size = 3, output_depth = 64, name = 'layer2')
    pool_1 = max_pooling_layer(conv_1, 2)
    conv_2 = conv_layer(pool_1, conv_size = 3, output_depth = 128, name = 'layer3')
    conv_2 = conv_layer(conv_2, conv_size = 3, output_depth = 128, name = 'layer4')
    pool_2 = max_pooling_layer(conv_2, 2)
    conv_3 = conv_layer(pool_2, conv_size = 3, output_depth = 256, name = 'layer5')
    conv_3 = conv_layer(conv_3, conv_size = 3, output_depth = 256, name = 'layer6')
    pool_3 = max_pooling_layer(conv_3, 2)
    conv_4 = conv_layer(pool_3, conv_size = 3, output_depth = 512, name = 'layer7')
    conv_4 = conv_layer(conv_4, conv_size = 3, output_depth = 512, name = 'layer8')
    drop_4 = dropout_layer(conv_4, keep_prob)
    pool_4 = max_pooling_layer(drop_4, 2)
    
    conv_5 = conv_layer(pool_4, conv_size = 3, output_depth = 1024, name = 'layer9')
    conv_5 = conv_layer(conv_5, conv_size = 3, output_depth = 1024, name = 'layer10')
    drop_5 = dropout_layer(conv_5, keep_prob)

    uppool_6 = unpooling_layer(drop_5, 2)
    concat_6 = concat_layer(drop_4, uppool_6)
    deconv_6 = deconv_layer(concat_6, conv_size = 3, output_depth = 512, name = 'layer11')
    deconv_6 = deconv_layer(deconv_6, conv_size = 3, output_depth = 512, name = 'layer12')

    uppool_7 = unpooling_layer(deconv_6, 2)
    concat_7 = concat_layer(conv_3, uppool_7)
    deconv_7 = deconv_layer(concat_7, conv_size = 3, output_depth = 256, name = 'layer13')
    deconv_7 = deconv_layer(deconv_7, conv_size = 3, output_depth = 256, name = 'layer14')

    uppool_8 = unpooling_layer(deconv_7, 2)
    concat_8 = concat_layer(conv_2, uppool_8)
    deconv_8 = deconv_layer(concat_8, conv_size = 3, output_depth = 128, name = 'layer15')
    deconv_8 = deconv_layer(deconv_8, conv_size = 3, output_depth = 128, name = 'layer16')

    uppool_9 = unpooling_layer(deconv_8, 2)
    concat_9 = concat_layer(conv_1, uppool_9)
    deconv_9 = deconv_layer(concat_9, conv_size = 3, output_depth = 64, name = 'layer17')
    deconv_9 = deconv_layer(deconv_9, conv_size = 3, output_depth = 64, name = 'layer18')
    deconv_10 = deconv_layer(deconv_9, conv_size = 3, output_depth = 2, name = 'layer19')
    conv_11 = conv_layer(deconv_10, conv_size = 1, output_depth = 1, name = 'layer20',
        activation_function = tf.nn.sigmoid, normalize = False)

    return conv_11


def Mini_Unet(inputs, keep_prob = 1):
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

    return conv_6



class Network(object):
    def __init__(self, net_name, y_size, x_size, channels, classes, pretrain_path = None):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param inputs: Data to predict on. Shape [batch_size, x_size, y_size, channels]
        :returns prediction: The unet prediction Shape [labels] (px=nx-self.offset/2)
        """
        self.net_name = net_name
        self.inputs = tf.placeholder(tf.float32, shape = [None, y_size, x_size, channels], name = 'input') 
        self.ground_truth = tf.placeholder(tf.int32, shape = [None, y_size, x_size, classes], 
            name = 'ground_truth')
        self.keep_prob = tf.placeholder(tf.float32, name = 'dropout_probability')
        self.parameter = []
        self.pretrain_path = pretrain_path
        self.sess = tf.Session()

        with tf.name_scope('result'):
            if net_name == 'Unet':
                self.predicter = Unet(self.inputs, self.keep_prob)
            elif net_name == 'Mini_Unet':
                self.predicter = Mini_Unet(self.inputs, self.keep_prob)
            elif net_name == 'VGGUnet':
                self.predicter = VGGUnet(self.inputs, self.keep_prob, self.parameter)

        with tf.name_scope('binary_entropy'):
            self.binary_entropy = image_binary_entropy(self.predicter, self.ground_truth)

        with tf.name_scope('dice_loss'):
            self.dice_loss = image_dice_loss(self.predicter, self.ground_truth)

        with tf.name_scope('focal_loss'):
            self.focal_loss = image_focal_loss(self.predicter, self.ground_truth, 2)

        with tf.name_scope('mixed_loss'):
            self.mixed_loss = image_mixed_loss(self.predicter, self.ground_truth)

        self.saver = tf.train.Saver()


    def _initialize(self, optimizer = 'Adam', learning_rate = 1e-4, momentum = 0.2):
        # tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        if optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

        self.train_step = optimizer.minimize(self.mixed_loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.net_name == 'VGGUnet' and self.pretrain_path != None:
            Network.load_pretrain_parameter(self)


    def predict(self, inputs):
        """
        Uses the model to create a prediction for the given data

        :param sess: current session
        :param inputs: Data to predict on. Shape [batch_size, x_size, y_size, channels]
        :returns prediction: The unet prediction. Shape [batch_size, x_size, y_size, channels]
        """

        prediction = self.sess.run(self.predicter, feed_dict = {self.inputs: inputs, self.keep_prob: 1.})

        return prediction

    def train(self, inputs, ground_truth):
        """
        Uses the input & ground truth to train the model

        :param sess: current session
        :param inputs: Data to train. Shape [batch_size, x_size, y_size, channels]
        :param ground truth: Data to calculate loss and model cability. 
            Shape [batch_size, x_sizedeconv_layer, y_size, channels]
        :return loss: Batch binary cross entropy
        """

        self.sess.run(self.train_step, feed_dict = {self.inputs: inputs, self.ground_truth: ground_truth, 
            self.keep_prob: 0.5})

        loss = self.sess.run(self.mixed_loss, feed_dict = {self.inputs: inputs, self.ground_truth: ground_truth,
            self.keep_prob: 1})

        return loss


    def save(self, model_path = 'ckpt/'):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        save_path = self.saver.save(self.sess, model_path)
        return save_path


    def restore(self, model_path = 'ckpt/', epochs = None):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        if epochs == None:
            file_check_point = open(model_path + 'checkpoint')
            check_point = file_check_point.readlines()
            final_model = (check_point[0].strip('\n').split(' '))[1].strip('"')

            self.saver.restore(self.sess, model_path + final_model)
            print("Model restored from file: %s" % (model_path + final_model))
        else:
            model_name = model_path + self.net_name + '_model_' + str(epochs) + '.ckpt'
            
            print("Model restored from file: %s" % model_name)
            self.saver.restore(self.sess, model_name)


    def load_pretrain_parameter(self):
        weights = np.load(self.pretrain_path)
        keys = sorted(weights.keys())
        # print(self.sess.run(self.parameter[0]))
        print('\nLoad Pretrain Parameter: VGG16----------')
        for i, k in enumerate(keys):
            if k.strip("_0123456789bW") != "conv":
                # print(k)
                continue
            print( i, k, np.shape(weights[k]))
            # self.sess.run(tf.assign(self.parameter[i], weights[k]))
            self.sess.run(self.parameter[i].assign(weights[k]))
            # print(self.sess.run(self.parameter[0]))
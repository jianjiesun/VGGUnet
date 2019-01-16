import tensorflow as tf
import numpy as np


def weight_variable(shape, name, stddev = 0.1):
	initial = tf.truncated_normal(shape, stddev = stddev)
	weights = tf.Variable(initial, dtype = tf.float32, name = name)
	return weights


def bias_variable(shape, name, stddev = 0.1):
	initial = tf.constant(stddev, shape = shape)
	bias = tf.Variable(initial, dtype = tf.float32, name = name)
	return bias


def normalize_variable(shape, name):
	initial = tf.ones(shape = shape)
	variable = tf.Variable(initial, dtype = tf.float32, name = name)
	return variable

def nomalize_layer(input_data, name, training):
	y_size = input_data.get_shape()[1]
	x_size = input_data.get_shape()[2]
	output_depth = input_data.get_shape()[3]
	output_shape = [y_size, x_size, output_depth]

	mean_variable = normalize_variable(output_shape, name = name + '_mean')
	var_variable = normalize_variable(output_shape, name = name + '_var')

	if training:
		mean, var = tf.nn.moments(input_data, axes = [0])
		ema = tf.train.ExponentialMovingAverage(decay=0.5)
		ema_apply_op = ema.apply([mean, var]) 
		tf.control_dependencies([ema_apply_op])
		mean_variable.assign(tf.identity(mean))
		var_variable.assign(tf.identity(var))

	scale = tf.Variable(tf.ones(output_shape), name = name + '_scale')
	shift = tf.Variable(tf.zeros(output_shape), name = name + '_shift')
	epsilon = 0.001

	conv_normalize = tf.nn.batch_normalization(input_data, mean_variable, var_variable, shift, scale, epsilon)
	return conv_normalize


def conv_layer(input_data, conv_size, output_depth, name, 
	activation_function = tf.nn.relu, normalize = True, parameter = None):
	with tf.name_scope(name):
		input_depth = input_data.get_shape()[3]
		mask_shape = tf.stack([conv_size, conv_size, input_depth, output_depth])

		conv_weights = weight_variable(mask_shape, name = name + '_w')
		conv_biases = bias_variable([output_depth], name = name + '_b')

		if parameter != None:
			parameter += [conv_weights, conv_biases]

		convolution = tf.nn.conv2d(input_data, conv_weights, strides = [1, 1, 1, 1], 
			padding = 'SAME') + conv_biases

		if normalize:
			conv_normalize =tf.nn.l2_normalize(convolution, axis = [0])
			# conv_normalize = tf.layers.batch_normalization(convolution, training = training, name = name + '_norm')
			# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		else:
			conv_normalize = convolution

		conv_activation = activation_function(conv_normalize)
	return conv_activation


def deconv_layer(input_data, conv_size, output_depth, name, 
	activation_function = tf.nn.relu, normalize = True, stride = 1):
	with tf.name_scope(name):
		batch_shape = tf.shape(input_data)[0]
		y_size = input_data.get_shape()[1] * stride
		x_size = input_data.get_shape()[2] * stride
		input_depth = input_data.get_shape()[3]
		mask_shape = tf.stack([conv_size, conv_size, output_depth, input_depth])
		output_shape = tf.stack([batch_shape, y_size, x_size, output_depth])

		deconv_weights = weight_variable(mask_shape, name = name + '_w')
		deconv_biases = bias_variable([output_depth], name + '_b')

		deconvolution = tf.nn.conv2d_transpose(input_data, deconv_weights, output_shape, 
			strides=[1, stride, stride, 1], padding='SAME') + deconv_biases

		if normalize:
			deconv_normalize =tf.nn.l2_normalize(deconvolution, axis = [0])
			# deconv_normalize = nomalize_layer(deconvolution, name = name, training = training)
		else:
			deconv_normalize = deconvolution

		deconv_activation = activation_function(deconv_normalize)
	return deconv_activation


def fully_connect_layer(input_data, input_depth, output_depth, name, 
	activation_function = tf.nn.relu, normalize = True):
	with tf.name_scope(name):
		fcn_w = weight_variable([input_depth, output_depth], name + '_w')
		fcn_b = bias_variable([output_depth], name + '_b')

		fcn_fcn = tf.matmul(input_data, fcn_w) + fcn_b

		if normalize:
			fcn_normal = tf.nn.l2_normalize(fcn_fcn, axis = [0])
		else:
			fcn_normal = fcn_fcn

		fcn_activation = activation_function(fcn_normal)
	return fcn_activation


def max_pooling_layer(input_data, pool_size):
	output = tf.nn.max_pool(input_data, ksize = [1, pool_size, pool_size, 1], 
		strides = [1, pool_size, pool_size, 1], padding = 'SAME')
	return output


def unpooling_layer(input_data, unpool_size):
	layer_shape = input_data.get_shape()
	output_size = [layer_shape[1] * unpool_size, layer_shape[2] * unpool_size]
	output = tf.image.resize_nearest_neighbor(input_data, output_size)
	return output


def concat_layer(input_data_1, input_data_2):
	output = tf.concat([input_data_1, input_data_2], axis = 3)
	return output


def dropout_layer(input_data, keep_prob):
	output = tf.nn.dropout(input_data, keep_prob)
	return output


def image_binary_entropy(input_data, ground_truth):
	GT_float = tf.to_float(ground_truth)
	positive_entropy = -1 * tf.multiply(GT_float, tf.log(input_data))
	negative_entropy = -1 * tf.multiply((1 - GT_float), tf.log(1 - input_data))

	batch_entropy = tf.reduce_mean(positive_entropy + negative_entropy)
	return batch_entropy


def mean_square_error(input_data, ground_truth):
	GT_float = tf.to_float(ground_truth)
	square_error = tf.square(input_data - GT_float)
	image_total_error = tf.reduce_sum(square_error, reduction_indices = [1, 2, 3])
	mean_error = tf.reduce_mean(image_total_error)
	return mean_error


def image_dice_loss(input_data, ground_truth):
	GT_float = tf.to_float(ground_truth)
	product = tf.multiply(input_data, GT_float)

	intersection = tf.reduce_sum(product, reduction_indices = [1, 2, 3])
	GT_sum = tf.reduce_sum(GT_float, reduction_indices = [1, 2, 3])
	prediction_sum = tf.reduce_sum(input_data, reduction_indices = [1, 2, 3])

	coefficient = (2. * intersection + 1.) / (GT_sum + prediction_sum + 1.)
	loss = tf.reduce_mean(-1 * tf.log(coefficient))
	return loss


def image_focal_loss(input_data, ground_truth, gamma):
	GT_float = tf.to_float(ground_truth)
	positive_parameter = -1 * tf.pow(1 - input_data, gamma)
	positive_loss = tf.multiply(GT_float, tf.multiply(positive_parameter, tf.log(input_data)))

	negative_parameter = -1 * tf.pow(input_data, gamma)
	negative_loss = tf.multiply(1 - GT_float, tf.multiply(negative_parameter, tf.log(1 - input_data)))

	focal_loss = tf.reduce_mean(positive_loss + negative_loss)
	return focal_loss


def image_mixed_loss(input_data, ground_truth):
	loss = 10. * image_focal_loss(input_data, ground_truth, 2) + \
		image_dice_loss(input_data, ground_truth)
	return loss
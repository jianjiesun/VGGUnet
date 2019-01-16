from Model.model import *
from util import *
import numpy as np
from glob import glob
import cv2
from time import clock


# net parameter setting
Usage_net = "VGGUnet"
Image_x_size = 384
Image_y_size = 384
Image_channels = 3
Seg_classes = 1
epoches_num = 1
batch_size = 10


if __name__ == '__main__':
	# network initialize
	VGGUnet = Network(Usage_net, Image_y_size, Image_x_size, Image_channels, Seg_classes, pretrain_path = 'vgg16_weights.npz')
	VGGUnet._initialize(learning_rate = 1e-4)
	# VGGUnet.restore(model_path = 'ckpt/model_384_4epochs/', epochs = 4)

	#ground truth preprocess
	ground_truth, ground_truth_not_merge = ground_truth_read()
	total_file_name = list(enumerate(ground_truth))

	batch_number = int(len(ground_truth) / batch_size)
	print("batch_number:", batch_number)

	for epoch in range(epoches_num):
		print('Epoch %3d/%-3d' %(epoch + 1, epoches_num))
		loss_sum = 0
		for batch in range(batch_number): 

			input_image = train_image_data_open(total_file_name, Image_x_size, Image_y_size, batch_size, batch)
			input_label = train_GT_data_open(ground_truth, total_file_name, Image_x_size, Image_y_size, batch_size, batch)

			training_loss = VGGUnet.train(input_image, input_label)

			loss_sum = loss_sum + training_loss
			print_state(batch_number * batch_size, batch * batch_size, 'loss', training_loss, batch_size)

		print("epoch = " , epoch, "average loss = " ,loss_sum / batch_number)

		ckpt_name = "ckpt/" + Usage_net + "_model_" + str(epoch + 1) + ".ckpt"
		save_path = VGGUnet.save(ckpt_name)
		print("model save path:", save_path)

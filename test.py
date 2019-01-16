from Model.model import *
from util import *
from evaluation import *
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
batch_size = 10


if __name__ == '__main__':
	# network initialize
	VGGUnet = Network(Usage_net, Image_y_size, Image_x_size, Image_channels, Seg_classes)
	VGGUnet.restore(model_path = 'ckpt/')

	#ground truth preprocess
	ground_truth, ground_truth_not_merge = ground_truth_read()

	test_file_path = glob('test/*')

	total_TP = np.zeros(10)
	total_FN = np.zeros(10)
	total_FP = np.zeros(10)
	total_F2_score = 0

	for i in range(len(test_file_path)):
		input_image = test_image_open(ground_truth, test_file_path[i], Image_x_size, Image_y_size, batch_size)
		file_name = test_file_path[i].split('/')[7]
		print('\nfile name:', file_name)

		ground_truth_image = ground_truth_open(ground_truth, file_name)

		test_result = VGGUnet.predict(input_image)

		test_training_result = test_result[0]
		test_image = (test_training_result * 255).astype(np.uint8)
		test_image_resize = cv2.resize(test_image, (768, 768), interpolation = cv2.INTER_LINEAR)
		test_image_resize = test_image_resize.reshape(768, 768, 1)

		threshold_test_index = test_image_resize[:,:,:] > 127
		threshold_test = np.zeros((768, 768, 1), dtype = np.uint8)
		threshold_test[threshold_test_index] = 255

		TP, FN, FP = test_image_evaluation(ground_truth_not_merge, file_name, threshold_test)
		
		total_TP = total_TP + TP
		total_FN = total_FN + FN
		total_FP = total_FP + FP

		single_F2_score = F2_score(TP, FN, FP)
		
		total_F2_score = total_F2_score + single_F2_score

		print('True Positive :', TP)
		print('False Negative :', FN)
		print('False Positive :', FP)
		print('single F2 score:', single_F2_score)
		print('Total F2 score:', total_F2_score / (i + 1))

		cv2.namedWindow('original_image', cv2.WINDOW_NORMAL)
		cv2.namedWindow('threshold result', cv2.WINDOW_NORMAL)
		cv2.namedWindow('ground_truth', cv2.WINDOW_NORMAL)
		cv2.imshow('original_image', cv2.imread(test_file_path[i]))
		cv2.imshow('threshold result', threshold_test)
		cv2.imshow('ground_truth', ground_truth_image)
		if cv2.waitKey(0) == 27:
			break

from util import *
import numpy as np
from glob import glob
import cv2


def test_image_evaluation(ground_truth, file_name, test_image):
	label_number, labels = cv2.connectedComponents(test_image)
	label_number = label_number - 1

	total_single_label = []
	total_single_test = []
	iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	TP = np.zeros(len(iou_threshold))
	FP = np.zeros(len(iou_threshold))
	FN = np.zeros(len(iou_threshold))

	# if there have ship
	if file_name in ground_truth:
		ground_truth_number = len(ground_truth[file_name])

		# ground truth single ship image create
		for i in range(ground_truth_number):
			single_label_flat = np.zeros((768 * 768))

			for j in range(int(len(ground_truth[file_name][i]) / 2)):
				for k in range(int(ground_truth[file_name][i][j * 2 + 1])):
					single_label_flat[int(ground_truth[file_name][i][j * 2]) + k - 1] = 1

			single_label = single_label_flat.reshape((768, 768, 1)).swapaxes(1,0)

			total_single_label.append(single_label)

		# test single image create
		for i in range(1, label_number + 1):
			single_test = np.zeros((768, 768))
			single_test[labels == i] = 1
			single_test = single_test.reshape(768, 768, 1)

			total_single_test.append(single_test)

		# pair ground truth and test
		GT_index = np.ones((len(iou_threshold), ground_truth_number))
		test_index = np.ones((len(iou_threshold), label_number))

		for i in range(ground_truth_number):
			for j in range(label_number):

				intersection = np.sum(np.multiply(total_single_label[i], total_single_test[j]))
				ground_truth_region = np.sum(total_single_label[i])
				test_region = np.sum(total_single_test[j])

				iou = intersection / (ground_truth_region + test_region - intersection)

				for k in range(len(iou_threshold)):
					if iou >= iou_threshold[k]:
						GT_index[k, i] = 0
						test_index[k, j] = 0
						TP[k] = TP[k] + 1

		FN = np.sum(GT_index, axis = 1)
		FP = np.sum(test_index, axis = 1)

		return TP, FN, FP

	else:
		TP = np.zeros(len(iou_threshold))
		FP = np.ones(len(iou_threshold)) * label_number
		FN = np.zeros(len(iou_threshold))

		return TP, FN, FP

def F2_score(TP, FN, FP):
	beta = 2
	iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

	if np.sum(TP + FN + FP) != 0:
		F_beta = (1 + np.power(beta, 2)) * TP / ((1 + np.power(beta, 2)) * TP + np.power(beta, 2) * FN + FP)
		score = np.sum(F_beta) / len(iou_threshold)
	else:
		score = 1

	return score
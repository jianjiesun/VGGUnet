import numpy as np
from glob import glob
import cv2
import random


def ground_truth_read():
	# input image & ground truth load
	file_ground_truth = open(('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/airbus ship/'
		'all/train_ship_segmentations_v2.csv'))
	ground_truth_data = np.array(file_ground_truth.readlines())


	# ground truth label split
	# only positive sample
	GT_dict_merge = {}
	GT_dict_not_merge = {}
	positive_file_name = []
	print('Ground Truth processing----------')
	for i in range(1, len(ground_truth_data)):
		file_name = ground_truth_data[i].strip('\n').split(',')[0]
		file_data = (ground_truth_data[i].strip('\n').split(',')[1]).split(' ')
		file_data_list = np.copy(file_data)
		
		if file_data[0] != '':
			if file_name in GT_dict_merge:
				GT_dict_merge[file_name].extend(file_data)
				GT_dict_not_merge[file_name].extend([file_data_list])			
			else:
				GT_dict_merge[file_name] = file_data
				GT_dict_not_merge[file_name] = [file_data_list]
				try:
					positive_file_name.append(file_name)
				except:
					positive_file_name = file_name

		print_state(len(ground_truth_data), i, 'file name', file_name)

	print('Ground Truth processing Done----------')

	return GT_dict_merge, GT_dict_not_merge


def test_image_open(ground_truth, test_image_path, Image_x_size, Image_y_size, batch_size):
	batch_image = []
	total_file_name = list(enumerate(ground_truth))

	file_name = test_image_path.split('/')[7]
	path_image = ('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/airbus ship/all/train_v2/' + 
		file_name)

	single_image = cv2.imread(path_image)
	single_image_resize = cv2.resize(single_image, (Image_x_size, Image_y_size), interpolation = cv2.INTER_NEAREST)
	batch_image.append(single_image_resize)

	for i in range(batch_size - 1):
		random_number = int(random.random() * len(ground_truth))
		file_name = total_file_name[random_number][1]
		path_image = ('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/airbus ship/all/train_v2/' + 
			file_name)
		single_image = cv2.imread(path_image)
		single_image_resize = cv2.resize(single_image, (Image_x_size, Image_y_size), interpolation = cv2.INTER_NEAREST)
		batch_image.append(single_image_resize)

	return np.array(batch_image)


def train_image_data_open(total_file_name, Image_x_size, Image_y_size, batch_size, batch_number):
	batch_image = []

	start_number = batch_number * batch_size
	end_number = start_number + batch_size

	for i in range(start_number, end_number):
		file_name = total_file_name[i][1]
		try: 
			path_image = ('/media/jianjie/cf360de1-1e04-496f-aca5-083f49b831bb/airbus ship/all/train_v2/' + 
				file_name)
			single_image = cv2.imread(path_image)
		except:
			print(file_name, end = '')
			print('file doesn`t exist')
			continue

		single_image_resize = cv2.resize(single_image, (Image_x_size, Image_y_size), interpolation = cv2.INTER_LINEAR)

		batch_image.append(single_image_resize)

	return np.array(batch_image)

def train_GT_data_open(ground_truth, total_file_name, Image_x_size, Image_y_size, batch_size, batch_number):
	batch_label = []
	start_number = batch_number * batch_size
	end_number = start_number + batch_size

	for i in range(start_number, end_number):
		file_name = total_file_name[i][1]
		single_label = ground_truth_open(ground_truth, file_name)

		single_label_resize = cv2.resize(single_label, (Image_x_size, Image_y_size), interpolation = cv2.INTER_NEAREST)
		single_label_resize = single_label_resize.reshape((Image_x_size, Image_y_size, 1))
		
		batch_label.append(single_label_resize)

	return np.array(batch_label)



def ground_truth_open(ground_truth, file_name):
	single_label_flat = np.zeros((768 * 768))

	try:
		for j in range(int(len(ground_truth[file_name]) / 2)):
			for k in range(int(ground_truth[file_name][j * 2 + 1])):
				single_label_flat[int(ground_truth[file_name][j * 2]) + k - 1] = 1

		single_label = single_label_flat.reshape((768, 768, 1)).swapaxes(1,0)

	except:
		single_label = np.zeros((768,768,1))

	return single_label


def print_state(total_state, current_state, variable_name, variable, state_grid = 1):

	print('%d/%d'%(current_state + state_grid, total_state), '[', end = '')
	for j in range(25):

		if ((current_state + state_grid) / total_state) >= ((j + 1) / 25):
			print('=', end = '')
		else:
			print('.', end = '')

	print(']', variable_name, ':', variable, end = '\r')

	if current_state == total_state - state_grid:
			print(end = '\n')

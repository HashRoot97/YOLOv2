import tensorflow as tf
import numpy as np
import os
import sys
import threading
import random
from datetime import datetime
from operator import itemgetter

from IPython.display import clear_output
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2



tf.app.flags.DEFINE_string('data_csv', './../data.csv', 'CSV file')
tf.app.flags.DEFINE_string('anchor_boxes', './../anchors.txt', 'Anchor boxes dimentions')
tf.app.flags.DEFINE_string('output_directory', './../TFRecords', 'Output directory containing tfrecord files')

tf.app.flags.DEFINE_integer('num_threads', 4, 'Number of threads for processing tfrecords')
tf.app.flags.DEFINE_integer('num_shards', 10, 'Number of shards for training data')
tf.app.flags.DEFINE_integer('num_classes', 20, 'Number of classes')
tf.app.flags.DEFINE_integer('num_grids', 13, 'Number of grids')
tf.app.flags.DEFINE_integer('grid_w', 32, 'Width of the grid')
tf.app.flags.DEFINE_integer('grid_h', 32, 'Height of the grid')


classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}


FLAGS = tf.app.flags.FLAGS

if not (os.path.exists(FLAGS.output_directory)):
	os.mkdir(FLAGS.output_directory)

def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_anchors(file_path):
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			w, h = line.split()
			anchor = [float(w), float(h)]
			anchors.append(anchor)

	return np.asarray(anchors)



def read_csv(file_path):

	data = pd.read_csv(file_path)
	
	filenames = data['ImageLOC'][:]
	a = data['ROI'][:]
	for i in range(data['ROI'].shape[0]):
		b = a[i].replace('[', '')
		b = b.replace(']', '')
		b = np.fromstring(b, dtype=int, sep=' ')
		l = len(b)
		l = l/4
		b = np.split(b, l)
		data['ROI'][i] = np.asarray(b)
	rois = data['ROI']

	a =data['Classes'][:]
	for i in range(data['Classes'].shape[0]):
		b = a[i].replace('[', '')
		b = b.replace(']', '')
		b = b.replace("'", "")
		b = b.replace('\n', '')
		b = b.split(' ')
		data['Classes'][i] = np.asarray(b)
	classes = data['Classes']

	return filenames, rois, classes



def process_tfrecord(name, file_names, rois, classes, num_shards, anchors, num_anchors):

	assert len(file_names) == len(rois)
	assert len(rois) == len(classes)

	""" Diving all the images into different ranges"""
	spacing = np.linspace(0, len(file_names), FLAGS.num_threads+1).astype(np.int)
	ranges = []
	for i in range(len(spacing)-1):
		ranges.append([spacing[i], spacing[i+1]])

	print("Launching %d threads for spacings: %s" % (FLAGS.num_threads, ranges))

	coord = tf.train.Coordinator() # for coordinating all the threads

	threads = []

	for thread_idx in range(len(ranges)):
		args = (name, thread_idx, ranges, file_names, classes, rois, num_shards, FLAGS.num_threads, anchors, num_anchors)
		t = threading.Thread(target=process_tfrecord_batch, args=args)
		t.start()
		threads.append(t)

	# Wait for all threads to finish
	coord.join(threads)
	print("%s: Finished writing all %d images in dataset" %(datetime.now(), len(file_names)))



def print_image(roi, clas, image_data, filename=None, anchors=None):
			
			print(roi, clas, filename, sep=' ')
			fix, ax = plt.subplots()
			ax.imshow(image_data)
			for j in range(roi.shape[0]):
				x1 = roi[j][0]
				y1 = roi[j][1]
				x2 = roi[j][2]
				y2 = roi[j][3]
				#print(x1, y1, x2, y2, sep=' ')
				rect = patches.Rectangle((x1,y1), (x2-x1), (y2-y1), linewidth=2, edgecolor='g', facecolor='none')
				ax.add_patch(rect)
			plt.xticks([], [])
			plt.yticks([], [])
			plt.title(filename)
			plt.show()


def process_tfrecord_batch(name, thread_index, ranges, file_names, classes, rois, num_shards, num_threads, anchors, num_anchors):
	
	"""Processes and saves list of images as TFRecord in 1 thread."""
	num_shards_per_batch = int(num_shards/num_threads)
	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch+1).astype(int)
	num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	for s in range(num_shards_per_batch):
		shard = thread_index * num_shards_per_batch + s
		output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
		output_file = os.path.join(FLAGS.output_directory, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_count = 0
		files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
		

		for i in files_in_shard:
			filename = file_names[i]
			clas = classes[i]
			roi = rois[i]

			image_data = _process_image(filename)
			
			#print_image(roi, clas, image_data, filename, anchors)
			
			label = create_labels(roi, clas, num_anchors, anchors)
			
			example = convert_to_example(image_data, label)
			
			writer.write(example.SerializeToString())
			shard_count += 1
			counter += 1

		
		writer.close()
		print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_count, output_file))
		shard_count = 0
	print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, counter, num_files_in_thread))



def _process_image(filename):

	"""Read image files from disk"""
	img_data = cv2.imread(filename)
	img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

	return img_data



def create_labels(rois, classes, num_anchors, anchors):

	labels = np.zeros([FLAGS.num_grids, FLAGS.num_grids, (num_anchors*(5 + FLAGS.num_classes))])
	
	for roi, clas in zip(rois, classes):

		grid_x, grid_y = get_grid_cell(roi, FLAGS.grid_w, FLAGS.grid_h)
		active_anchors = get_active_anchors(roi, anchors, grid_x, grid_y)
		active_anchors.sort(key=itemgetter(1), reverse=True)
		

		for i in range(len(active_anchors)):

			anchor_label =  anchor_to_label(roi, anchors, active_anchors[i], grid_x, grid_y, clas)
			if type(anchor_label) == int:
				del(active_anchors[i])
			else:
				labels[grid_x, grid_y, :] += anchor_label[grid_x, grid_y, :]
				break

	return labels



def get_active_anchors(roi, anchors, grid_x, grid_y):
	
	anchors = redef_anchors(anchors, [grid_x, grid_y])
	indxs = []
	iou_max, idx_max = 0, 0

	for i in range(5):
		iou = get_iou(roi, anchors[i])
		
		#if iou>FLAGS.iou_threshold:
		#	indxs.append([i, iou])
		indxs.append([i, iou])
				
		#if iou>iou_max:
		#	iou_max, idx_max = iou, i

	#if(len(indxs)==0):
	#	indxs.append(idx_max)


	return indxs





def redef_anchors(anchors_, grid):
    
    anchors = []
    grid_x = grid[0]*32
    grid_y = grid[1]*32
    mid_grid = (grid_x + (32/2), grid_y + (32/2))
    for i in range(5):
        anc_w = anchors_[i][0]*32
        anc_h = anchors_[i][1]*32
        anc_x = mid_grid[0] - (anc_w/2)
        anc_y = mid_grid[1] - (anc_h/2)
        anc_x_max = anc_x + anc_w
        anc_y_max = anc_y + anc_h
        anc = [int(anc_x), int(anc_y), int(anc_x_max), int(anc_y_max)]
        
        for i in range(len(anc)):
            if anc[i] < 0:
                anc[i] = 0
            elif anc[i] > 416:
                anc[i] = 416
        anchors.append(anc)

    return anchors




def get_iou(bb1, bb2):

	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def get_grid_cell(roi, grid_w, grid_h):

	x_center = roi[0]+(roi[2]-roi[0])/2.0 
	y_center = roi[1]+(roi[3]-roi[1])/2.0 

	grid_x = int(x_center/(float(grid_w)))
	grid_y = int(y_center/(float(grid_h)))
	
	return grid_x, grid_y



def anchor_to_label(roi, anchors, active_anchors, grid_x, grid_y, clas):

	label = np.zeros([FLAGS.num_grids, FLAGS.num_grids, (5*(5 + FLAGS.num_classes))])

	# if active_anchors[0] != 0:
	# 	j = 25*active_anchors[0]-1
	# else:
	j = 25*active_anchors[0]


	if label[grid_x, grid_y, j] != 0:
		
		return 0
	else:

		label[grid_x, grid_y, j] = 1.0

		for i in range(4):
			label[grid_x, grid_y, j+i+1] = roi[i]
		
		label[grid_x, grid_y, j+5+classes_num[clas]] = 1

		return label
	
	#print("Label for anchor %d" %active_anchors[0], label[grid_x, grid_y, :])



	

def convert_to_example(image_data, labels):

	 """Convert the values to Tensotflow TFRecord example for saving in the TFRecord file"""
	 img_raw = image_data.tostring()
	 labels_raw = labels.tostring()
	 example = tf.train.Example(features=tf.train.Features(feature={
	 	'image_data': bytes_feature(img_raw),
	 	'labels': bytes_feature(labels_raw)
	 	}))
	 return example



def make_tfrecord(name):

	#print(FLAGS)
	print('Reading anchor.txt.....')
	anchors = read_anchors(FLAGS.anchor_boxes)
	num_anchors = anchors.shape[0]
	print('Number of anchors in anchor.txt: %d' % num_anchors)
	
	print('Reading data.csv.....')
	csv_filenames, csv_rois, csv_classes = read_csv(FLAGS.data_csv)

	process_tfrecord(name, csv_filenames, csv_rois, csv_classes, FLAGS.num_shards, anchors, num_anchors)

	label = np.zeros([FLAGS.grid_w, FLAGS.grid_h, num_anchors, FLAGS.num_classes], dtype=np.float32)




if __name__ == '__main__':

	print('Saving results to %s' % FLAGS.output_directory)
	make_tfrecord('train')
import tensorflow as tf
import numpy as np
import os
import sys
import threading
import random
from datetime import datetime

from IPython.display import clear_output
import cv2
import pandas as pd

csv_path = './data.csv'
anchor_boxes = './anchor.txt'
tfrecord_path = './TFRecords'


tf.app.flags.DEFINE_string('data_csv', './data.csv', 'CSV file')
tf.app.flags.DEFINE_string('anchor_boxes', './anchors.txt', 'Anchor boxes dimentions')
tf.app.flags.DEFINE_string('output_directory', './TFRecords', 'Output directory containing tfrecord files')

tf.app.flags.DEFINE_integer('num_threads', 10, 'Number of threads for processing tfrecords')
tf.app.flags.DEFINE_integer('shards', 10, 'Number of shards for training data')


FLAGS = tf.app.flags.FLAGS

if !(os.path.exists(FLAGS.output_directory)):
	os.mkdir(FLAGS.output_directory)

def int64_feature(value):
	if not isinstace(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def bytes_value(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_anchors(file_path):
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			anchors.append(map(float, line.split()))

	return np.array(anchors)



def read_csv(file_path):

	data = pd.read_csv(file_path)
	filenames = data['file names']
	rois = data['roi']
	classes =data['classes']

	return filenames, rois, classes



def process_images(name, file_names, rois, classes, num_shards):

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
		args = (name, thread_idx, ranges, file_names, classes, rois, num_shards, FLAGS.num_threads)
		t = threading.Thread(target=process_images_batch, args=args)
		t.start()
		threads.append(t)

	# Wait for all threads to finish
	coord.join(threads)
	print("%s: Finishes writing all %d images in dataset" %(datetime.now(), len(file_names)))



def process_image(filename):

	"""Read image files from disk"""
	pass


def convert_to_example(filename, iamge_data, height, width, roi, clas):

	 """Convert the values to Tensotflow TFRecord example for saving in the TFRecord file"""
	 pass



def process_images_batch(name, thred_index, ranges, file_names, classes, rois, num_shards, num_threads):
	
	"""Processes and saves list of images as TFRecord in 1 thread."""
	num_shards_per_batch = int(num_shards/num_threads)
	shard_ranges = (ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch+1).astype(int)
	num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	for s in range(num_shards_per_batch):
		shard = thread_index * num_shards_per_batch + s
		output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
		output_file = os.path.join(FLAGS.output_directory, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_count = 0
		files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
		

		for i in range(files_in_shard):
			filename = file_names[i]
			clas = classes[i]
			roi = rois[i]

			image_data, height, width = process_image(filename)
			example = convert_to_example(filename, image_data, height, width, roi, clas)
			
			writer.write(example.SerializeToString())
			shard_count += 1
			counter += 1

		
		writer.close()
		print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_counter, output_file))
		shard_count = 0
	print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, counter, num_files_in_thread))




def make_tfrecord():

	#print(FLAGS)
	print('Reading anchor file...')
	anchors = read_anchors(FLAGS.anchor_boxes)
	num_anchors = anchors.shape[0]
	print(num_anchors)
	
	
	csv_filenames, csv_rois, csv_classes = read_csv(FLAGS.data_csv)

	process_image(csv_filenames, csv_rois, csv_classes, FLAGS.num_shards)



if __name__ == '__main__':

	print('Saving results to %s' % FLAGS.output_directory)
	make_tfrecord('train')
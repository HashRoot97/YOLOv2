import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches




classes_name = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus',
     6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog',
    12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep',
    17: 'sofa', 18: 'train', 19: 'tvmonitor'}




def read_anchors(file_path='./../anchors.txt'):
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			w, h = line.split()
			anchor = [float(w), float(h)]
			anchors.append(anchor)

	return np.asarray(anchors)


def list_tfrecord_file(folder, file_list):

    
	tfrecord_list = []
	for i in range(len(file_list)):
		current_file_abs_path = os.path.join(folder, file_list[i])		
		if current_file_abs_path.endswith(".tfrecord"):
			tfrecord_list.append(current_file_abs_path)
			print("Found %s successfully!" % file_list[i])				
		else:
			pass
	return tfrecord_list


	
# Traverse current directory
def tfrecord_auto_traversal(folder, current_folder_filename_list):

	if current_folder_filename_list != None:
		print("%s files were found under %s folder. " % (folder, len(current_folder_filename_list)))
		print("Please be noted that only files ending with '*.tfrecord' will be loaded!")
		tfrecord_list = list_tfrecord_file(folder, current_folder_filename_list)
		if len(tfrecord_list) != 0:
			print("Found %d files:\n %s\n\n\n" %(len(tfrecord_list), current_folder_filename_list))
		else:
			print("Cannot find any tfrecord files, please check the path.")
	return tfrecord_list



def read_tf_records(batch_size=32):

	current_folder_filename_list = os.listdir("./../TFRecords")
	tf_records = tfrecord_auto_traversal("./../TFRecords", current_folder_filename_list)

	with tf.Session() as sess:
		features = {'image_data': tf.FixedLenFeature([], tf.string),
					'labels': tf.FixedLenFeature([], tf.string)}

		min_queue_examples = 500

		for i in range(len(tf_records)):
			filename_queue = tf.train.string_input_producer([tf_records[i]], num_epochs=None)

			reader = tf.TFRecordReader()
			_, serialized_example = reader.read(filename_queue)


			batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, capacity=min_queue_examples+100*batch_size, num_threads=1, 
				min_after_dequeue=min_queue_examples)
			parsed_example = tf.parse_example(batch, features=features)

			image_raw = tf.decode_raw(parsed_example['image_data'], tf.uint8)
			image = tf.cast(tf.reshape(image_raw, [batch_size, 416, 416, 3]), tf.float64)
			image = image/255.0

			labels_raw = tf.decode_raw(parsed_example['labels'], tf.float64)
			label = tf.reshape(labels_raw, [batch_size, 13, 13, 125])
			
			

		init_op = tf.group(tf.global_variables_initializer(), tf.global_variables_initializer())

		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		print('----------------------------------------------\n                Running\n----------------------------------------------')
		anchors = read_anchors()
		for i in range(20):

			img, lbl = sess.run([image, label])
			fix, ax = plt.subplots()
			ax.imshow(img[0])
			roi = []
			classes = []

			for x in range(13):
				for y in range(13):
					for z in range(5):
						if lbl[0, x, y, 25*z] != 0:
							r = np.zeros((4,))
							r[0] = lbl[0, x, y, 25*z+1] * 416.0
							r[1] = lbl[0, x, y, 25*z+2] * 416.0

							r[2] = lbl[0, x, y, 25*z+3] * (anchors[z][0]*32)
							r[3] = lbl[0, x, y, 25*z+4] * (anchors[z][1]*32)

							roi.extend(r)
							for t in range(20):
								if lbl[0, x, y, 25*z+5+t] != 0:
									classes.append(t)
			
			print(roi, len(roi), classes)
			q = p = 0
			while q<len(roi):

				x = roi[q]
				y = roi[q+1]
				x_max = roi[q+2]
				y_max = roi[q+3]

				rect = patches.Rectangle((x, y), (x_max), (y_max), linewidth=2, edgecolor='r', facecolor='none')
				ax.add_patch(rect)
				ax.text(x, y, classes_name[classes[p]], horizontalalignment='left', verticalalignment='bottom', color='b')

				q += 4
				p += 1

			plt.xticks([], [])
			plt.yticks([], [])
			plt.show()

		coord.request_stop()

		coord.join(threads)





def main():
	read_tf_records(1)

if __name__ == "__main__":
	main()
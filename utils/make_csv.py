import os
from xml.etree import ElementTree as ET
import csv


def get_dict(ann_dir):

	dict_main = {
		'file_path' : [],
		'classes' : [],
		'roi' : []
	}
	img_source_path = './../Dataset/Train/VOCdevkit/VOC2007/JPEGImages/'

	for file in os.listdir(ann_dir):
		file = ann_dir + file
		tree = ET.parse(file)

		root = tree.getroot()
		classes = []
		bbox = []
		for child1 in root:
		     if child1.tag == 'object':
		             a = []
		             for child2 in child1:
		                     if child2.tag == 'name':
		                             classes.append(child2.text)
		                     elif child2.tag == 'bndbox':
		                             for child3 in child2:
		                                     a.append(child3.text)
		             bbox.append(a)
		     if child1.tag == 'filename':
		     	img_name = child1.text

		dict_main['classes'].append(classes)
		dict_main['roi'].append(bbox)
		dict_main['file_path'].append(img_source_path + img_name)
	os.system('touch ./../data.csv')
	csv_file = './../data.csv'

	with open(csv_file, 'w') as f:

		for i in range(len(dict_main['file_path'])):
			a = '"{}"'.format(str(dict_main['file_path'][i]))
			b = '"{}"'.format(str(dict_main['roi'][i]))
			c = '"{}"'.format(str(dict_main['classes'][i]))

			line = a + ',' + b + ',' + c + '\n'
			f.write(line)

	return dict_main


ann_dir = './../Dataset/Train/VOCdevkit/VOC2007/Annotations/'
dict_main = get_dict(ann_dir)
print(dict_main.keys())
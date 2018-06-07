import os
from xml.etree import ElementTree as ET



def get_dict(ann_dir):

	dict_main = {
		'filename' : [],
		'classes' : [],
		'bbox' : []
	}

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

		dict_main['classes'].append(classes)
		dict_main['bbox'].append(bbox)
		dict_main['filename'].append(file)
	return dict_main


ann_dir = './../Dataset/Train/VOCdevkit/VOC2007/Annotations/'
dict_main = get_dict(ann_dir)

print(len(dict_main['classes']))
print(len(dict_main['bbox']))
print(len(dict_main['filename']))
import cv2
from PIL import Image
import numpy as np

def pre_process(dict_main):
	print('Preprocessing..')
	for i in range(len(dict_main['classes'])):
		loc = dict_main['file_path'][i]
		r = dict_main['roi'][i]
		new_loc, new_r = resize_img(loc, r, i)
		dict_main['roi'][i] = new_r
		dict_main['file_path'][i] = new_loc

		if i%500 == 0:
			print(str(i) + ' Images Done')
	return(dict_main)



def resize_img(image_loc, roi, i):
    im = np.array(Image.open(image_loc), dtype=np.uint8)
    img = cv2.resize(im, (200, 200))
    new_image_loc = './../Dataset/Train/VOCdevkit/VOC2007/ResizedJPEGImages/' + str(i) + '.jpg'
    cv2.imwrite(new_image_loc, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    new_roi = []
    for i in range(roi.shape[0]):
        
        x_ = im.shape[1]
        y_ = im.shape[0]
        x_min_, y_min_, x_max_, y_max_ = (roi[i][0], roi[i][1], roi[i][2], roi[i][3])
        x_scale = (200 / y_)
        y_scale = (200 / x_)
        x_min = int(np.round(x_min_ * y_scale))
        y_min = int(np.round(y_min_ * x_scale))
        x_max = int(np.round(x_max_ * y_scale))
        y_max = int(np.round(y_max_ * x_scale))
        new_roi.append([x_min, y_min, x_max, y_max])
    new_roi = np.array(new_roi)
    return new_image_loc, new_roi
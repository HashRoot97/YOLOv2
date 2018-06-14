import os
import urllib.request
import sys

def _print_download_progress(count, block_size, total_size):
    
    pct_complete = float(count * block_size) / total_size

    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_extract():
	check_dirs = ['./../Dataset/Train', './../Dataset/Test']
	if not (os.path.exists(check_dirs[0]) and os.path.exists(check_dirs[1])):
		print('Extracting Files...')
		os.system('cd ../ && mkdir Dataset && cd Dataset && mkdir Train && mkdir Test')
		os.system('tar xf ./../VOCtrainval_06-Nov-2007.tar -C ./../Dataset/Train')
		os.system('tar xf ./../VOCtest_06-Nov-2007.tar -C ./../Dataset/Test')
		print('Extraction Complete')
	else:
		print('Files already extracted')


def maybe_download():
	files = ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar']
	url_train = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
	url_test = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar' 
	parent_dir = os.listdir('./../')
	if files[0] not in parent_dir and files[1] not in parent_dir:
		print('Downloading Files...')
		file_path, _ = urllib.request.urlretrieve(url=url_train,
												  filename='./../'+files[0],
												  reporthook=_print_download_progress)
		print()
		print('Train Dataset Download Finsihed')
		file_path, _ = urllib.request.urlretrieve(url=url_test,
												  filename='./../'+files[1],
												  reporthook=_print_download_progress)
		print()
		print('Test Dataset Download Finsihed')
	else:
		print('Files already downloaded')
	maybe_extract()

maybe_download()






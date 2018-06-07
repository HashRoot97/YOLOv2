# YOLOv2 implementation from scratch using Tensorflow
Implementation of YOLOv2 for real time object detection.

1. anchors.txt contains the dimentions of the anchor boxes.
2. data.csv contains metadata having 3 columns: ImageLoc , ROIs, Classes.
   Each row of Data.csv contains: path_to_the_image_on_disk, [xmin, ymin, height, width], classes


Steps for running the model:

1. Open terminal
2. pip install requirements.txt --> for python 2.x |or| pip3 install requirements.txt --> for python 3.x
3. python maybe_download_and_extract.py 
4. python make_tfrecord.py
5. Run jupyter notebook using command: jupyter-notebook and launch model.ipynb

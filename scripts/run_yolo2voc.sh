#!/bin/bash
# About: script for converting GT annotation from yolo format (normalized format with maxHeight and maxWidth)
# to COCO format (non-normalized)
# contact: sharib.ali@eng.ox.ac.uk

# change me
DATA_DIR1=/home/sven/Documents/EAD2019-master/data/EAD2020-Phase-II-Detection_Segmentation-99Frames/originalImages/

DATA_DIR2=/home/sven/Documents/EAD2019-master/data/EAD2020-Phase-II-Detection_Segmentation-99Frames/bbox/
#-------
BASE_FOLDER=../fileFormatConverters
CLASS_NAMES=../endo.names

if [ -d "$RESULT_FOLDER" ]; then rm -Rf $RESULT_FOLDER; fi

# change me (if you want to rename the folder to something else)
#RESULT_FOLDER=../csvFiles
RESULT_FOLDER=/home/sven/Documents/EAD2019-master/data/EAD2020-Phase-II-Detection_Segmentation-99Frames/yolo2voc/

# first make sure the list of files exist in your dataFolder
ext='.jpg'
j=0

echo "convert to VOC format..."

python $BASE_FOLDER/any2voc.py -baseImgFolder $DATA_DIR1 -baseBoxFolder $DATA_DIR2 -pathToClassNames $CLASS_NAMES  -outFolder $RESULT_FOLDER -datatype 'GT'



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:50:49 2018

@author: ead2019
"""


def read_txt_file(txtfile):
    import numpy as np
    lines = []

    with open(txtfile, "r") as f:
        for line in f:
            line = line.strip()
            lines.append(line)

    return np.array(lines)


def read_boxes(txtfile):
    import numpy as np
    lines = []

    with open(txtfile, "r") as f:
        for line in f:
            line = line.strip()
            box = np.hstack(line.split()).astype(np.float)
            box[0] = int(box[0])
            lines.append(box)

    return np.array(lines)


def read_obj_names(textfile):
    classnames = []

    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line) > 0:
                classnames.append(line)

    return np.hstack(classnames)


debug = 0
useParsing = 0

if __name__ == "__main__":

    import numpy as np
    #    import pandas as pd
    import argparse
    import glob

    annotationImagePaths = ''
    classnames = ''


    bbox_base_path = '/home/sven/Documents/mmdetection/data/EAD2020_dataType_framesOnly/gt_bbox/'
    #
    classFileName = 'class_list.txt'
    classtextfile = classFileName
    classnames = read_obj_names(classtextfile)
    annotationImagePaths = bbox_base_path


    # histogram
    ext = ['*.txt']
    classCounter = [0, 0, 0, 0, 0, 0, 0, 0]
    print(annotationImagePaths)
    for filename in sorted(glob.glob(annotationImagePaths + ext[0], recursive=True)):
        if debug:
            print(filename)
            # 1) read
        bboxes = read_boxes(filename)
        # 2) loop throgh for class types in array
        for i in range(len(bboxes)):
            for j in range(len(classnames)):
                if j == int(bboxes[i][0]):
                    classCounter[j] = (int(classCounter[j]) + 1)
                    # percentage
    label = (classCounter / np.sum(classCounter)) * 100
    # plot            
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.arange(8)
    plt.bar(x, classCounter)
    plt.xticks(x, classnames)
    plt.xticks(rotation=45)
    for i in range(len(classCounter)):
        plt.text(-0.32 + i, classCounter[i] + 10, s=str(round(label[i], 2)), size=10, color='red', fontweight='bold')

    plt.show()

    fig.savefig('bboxclasses_test.jpg', bbox_inches='tight')

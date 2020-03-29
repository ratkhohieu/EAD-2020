#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:24:24 2018

@author: ead2019
"""
import os.path as osp
import glob
import os
from matplotlib.ticker import NullLocator
import pylab as plt
import numpy as np
import cv2
import seaborn as sns


def read_img(imfile):
    import cv2

    return cv2.imread(imfile)


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


def read_detect(txtfile):
    boxes = []
    scores = []
    labels = []
    with open(txtfile, "r") as f:
        for line in f:
            score = float(line.split()[1])
            box = np.hstack(line.split()[2:]).astype(np.uint32)
            label = line.split()[0]
            scores.append(score)
            boxes.append(box)
            labels.append(label)
    return boxes, scores, labels


def yolo2voc(boxes, imshape):
    import numpy as np
    m, n = imshape[:2]

    box_list = []
    for b in boxes:
        cls, x, y, w, h = b

        x1 = (x - w / 2.)
        x2 = x1 + w
        y1 = (y - h / 2.)
        y2 = y1 + h

        # absolute:
        x1 = x1 * n;
        x2 = x2 * n
        y1 = y1 * m;
        y2 = y2 * m

        box_list.append([cls, x1, y1, x2, y2])

    if len(box_list) > 0:
        box_list = np.vstack(box_list)
    print(box_list)
    return box_list


def plot_boxes(ax, boxes, labels):
    color_pal = sns.color_palette('hls', n_colors=len(labels))

    for b in boxes:
        cls, x1, y1, x2, y2 = b
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=2, color=color_pal[int(cls)])
        plt.text(x1, y1, s=classes[int(cls)], color="black",
                 verticalalignment="top",
                 # bbox={"color": color, "pad": 0},
                 )
    return []


def draw(img, boxes, scores, labels, classes, font_scale, text_color, filename):
    color_pal = sns.color_palette('hls', n_colors=16)
    for bbox, label, score in zip(boxes, labels, scores):
        bbox_int = bbox
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        id_label = label2id(label, classes)

        ar = (np.array(color_pal[id_label]) * 255).astype(np.uint8)
        a = [int(i) for i in ar]
        cv2.rectangle(img, left_top, right_bottom, a, 2)
        label_text = label
        text_color = color_pal[id_label]
        cv2.putText(img, label_text + str(score), (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, a)
    cv2.imwrite(filename, img)


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, read_img(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def read_obj_names(textfile):
    import numpy as np
    classnames = []

    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line) > 0:
                classnames.append(line)
    print(np.hstack(classnames))
    return np.hstack(classnames)


def label2id(label, classes):
    for i, x in enumerate(classes):
        if label == x: id = i
    return id


if __name__ == "__main__":
    """
    Example script to read and plot bounding box annotations (which are provided in <x,y,w,h> format)
    
    (x,y) - box centroid
    (w,h) - width and height of box in normalised. 
    """
    # from matplotlib.ticker import NullLocator
    # import pylab as plt
    # ids = list()
    # for line in open('train.txt'):
    #     ids.append(line.strip())
    # for index in range (0,2199):
    #     img_id = ids[index]
    #     print(ids[index])
    #     imgfile = '../annotationImages_and_labels/00001.jpg'
    #     bboxfile = '../annotationImages_and_labels/00001.txt'
    #     txtpath = osp.join('../annotationImages_and_labels/', '%s.txt')
    #     imgpath = osp.join('../annotationImages_and_labels/', '%s.jpg')
    #
    #     classfile = '../class_list.txt'
    #     classes = read_obj_names(classfile)
    #
    #     img = read_img(imgpath % img_id)
    #
    #     boxes = read_boxes(txtpath % img_id)
    #
    #         # convert boxes from (x,y,w,h) to (x1,y1,x2,y2) format for plotting
    #     boxes_abs = yolo2voc(boxes, img.shape)
    #
    #
    #     fig, ax = plt.subplots()
    #         #print(img.shape)
    #     ax.imshow(img)
    #     plot_boxes(ax, boxes_abs, classes)
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #
    #     plt.savefig(f"GT/{img_id}.jpg")

    # gtfolder = '/home/sven/Documents/EAD2019-master/data/EAD2020_dataType_framesOnly/frames/'
    #
    # # get a list with the ground-truth files
    # ground_truth_files_list = glob.glob(os.path.join(gtfolder, '*.jpg'))
    # for txt_file in ground_truth_files_list:
    #     filename = txt_file.replace('.jpg', '.txt')
    #
    #     filename = filename.replace('/frames','/pre')
    #     classfile = '../class_list.txt'
    #     classes = read_obj_names(classfile)
    #
    #     img = read_img(txt_file)
    #
    #     boxes = read_boxes(filename)
    #
    #     # convert boxes from (x,y,w,h) to (x1,y1,x2,y2) format for plotting
    #     boxes_abs = __txt2list__(boxes)
    #
    #     fig, ax = plt.subplots()
    #     # print(img.shape)
    #     ax.imshow(img)
    #     plot_boxes(ax, boxes_abs, classes)
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #
    #     namefig = txt_file.replace('/data/EAD2020_dataType_framesOnly/', '/examples/Pre1/')
    #     namefig
    #     plt.savefig(namefig)

    classfile = '../class_list.txt'
    classes = read_obj_names(classfile)


    root = '/home/sven/Documents/mmdetection/demo/'
    root_img='/home/sven/Documents/EAD2019-master/examples/'

    folders = ['ead2020/Detection/',
               'ead2020/Generalization/',
               'ead2020/Detection_sequence/']

    results = ['EndoCV2020_testSubmission/detection_bbox/',
               'EndoCV2020_testSubmission/generalization_bbox/',
               'EndoCV2020_testSubmission/sequence_bbox/']

    images_files_list = {}

    for id_result, folder in enumerate(folders):

        images_files_list = glob.glob(os.path.join(root + folder, '*.jpg'))
        #         print(images_files_list)

        buffer = osp.join(root + 'predict/final/' + results[id_result])
        buffer2 = osp.join('predict/final/' + results[id_result])
        # print(buffer)

        if not os.path.exists(buffer):  # if it doesn't exist already
            os.makedirs(buffer, exist_ok=True)

        for image_file in images_files_list:
            filename = '/home/sven/Documents/mmdetection/demo/predict/final/all1/' + image_file.replace('.jpg','.txt').split('/')[-1]
            print(filename)
            img = cv2.imread(image_file)
            boxes, score, label = read_detect(filename)
            filename=filename.replace('.txt','.jpg')
            print(filename)
            draw(img, boxes, score, label, classes, font_scale=0.5, text_color='black', filename=filename)
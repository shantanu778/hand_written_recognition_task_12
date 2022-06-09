from __future__ import print_function
import re
from statistics import mean, median
import math
import cv2 as cv
import numpy as np
import random as rng
from segmentations.line.line import *
import matplotlib.pyplot as plt


rng.seed(12345)

def segment_character(line, image_name, line_idx, verbose):
    # print(line.shape)

    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv.dilate(line, kernel, iterations=1)
    edges = cv.Canny(dilate_img, 100, 150, 3)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    #create an empty image for contours
    img_contours = np.zeros(edges.shape)

    cv.drawContours(img_contours, contours, -1, (255), 2)
    img_contours = np.uint8(img_contours)


    # Applying cv2.connectedComponents() 
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_contours)

    image_list = []
    diff = []
    ROI_number = 0
    for i in range(len(stats)):
        if i != 0:
            # print(stat)
            x = stats[i][0]
            width = stats[i][2]
            y = stats[i][1]
            height = stats[i][3]
            if width >= 6:
                ROI = line[y:y+height,x:x+width]
                image_list.append((ROI, x, width))
                diff.append(width)
            
            ROI_number += 1

    # print(diff)

    if len(image_list)==0 and len(diff)==0:
        return None, None

    image_list = sorted(image_list, key = lambda x: x[1])

    spaces = []
    for i in range(len(image_list)):
        space = image_list[i][1] - image_list[i-1][1]
        if space >= 30:
            spaces.append(i)

    # print(spaces)
    
    total_char_std = 12.12

    if len(diff)>0:
        avg_width = math.ceil(mean(diff))
        current_std = np.std(np.array(diff))

    else:
        current_std = total_char_std

    character_list = []

    if current_std > total_char_std:
        difference = np.percentile(np.array(diff),90)
        # print("difference", difference)
        for idx, item in enumerate(image_list):
            # print("item", item[-1])
            img = item[0]
            if not item[-1]>difference:
                character_list.append(img)
            else:
                counter = math.ceil(img.shape[-1]/avg_width)
                # print(img.shape[-1], counter)
                for i in range(counter):
                    split = img[:, avg_width*i :avg_width*(i+1)] 
                    # print(split.shape)
                    character_list.append(split)

    else:
        for idx, item in enumerate(image_list):
            character_list.append(item[0])

    if verbose:
        folder = f"characters/{image_name}/line_{line_idx}/"
        os.makedirs(folder, exist_ok = True)
        for idx, img in enumerate(character_list):
            path = f"{folder}/charater_{idx+1}.png"
            try:
                cv.imwrite(path, img)
            except:
                print("Empty Charset")
    
    return character_list, spaces


def segment_lines(src, verbose):

    img = cv.imread('dummy.jpg')

    lines, image_name, output_dir = segment(src, outputDir = 'lines_output', verbose=verbose)
    # lines.insert(3, img)

    print("Total Lines before Segmentation", len(lines))

    total_spaces = []
    line_chars = []
    for idx, line in enumerate(lines):
        # print(line)
        if line is not None:
            character_list, spaces = segment_character(line, image_name, idx, verbose)
            if character_list is not None and len(character_list)>0:
                line_chars.append(character_list)
                total_spaces.append(spaces)
            else:
                print("no chars detected")
                total_spaces.append([-100]) 
                pass
        else:
            print("no line detected")
            total_spaces.append([-100]) 
            pass

    print("Total Lines after Character Segmentation", len(line_chars))
    return line_chars, total_spaces




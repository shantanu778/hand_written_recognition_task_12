import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
import cv2
import os
import numpy as np 
from scipy.signal import find_peaks, peak_widths

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def read_image_list(pathToList):
    '''

    :param pathToList:
    :return:
    '''
    dirs = os.listdir(pathToList)
    imagePaths = [f'{pathToList}/{imageName}' for imageName in dirs]
    
    return imagePaths


def line_segment(imageName,originalImage,detectedLineImage, outputDir, threshold=25,verbose=False):

    
    lines = getLines(detectedLineImage)

    highlightedImage = copy.deepcopy(originalImage)
    highlightedImage = cv2.cvtColor(highlightedImage,cv2.COLOR_GRAY2RGB)
    total_lines = []
    for i,coord in enumerate(lines):
        
        low_y = coord[0][1]-threshold
        high_y = coord[-1][1]+10    
        max_h = max([cor[3] for cor in coord])
        high_y = max_h+high_y
        
        highlightedImage = cv2.rectangle(highlightedImage,(0,low_y),(originalImage.shape[1],high_y),(255,255,51),5)
        t = originalImage[low_y:high_y,:]
            
        t = clean_line(t)
        # total_lines.append(t)
        try:
            total_lines.append(t)
            cv2.imwrite(f"{outputDir}/{imageName}/line_{i+1}.png",t)
            
        except:
            print("empty lines")
    
    if verbose:
        cv2.imwrite(f"{outputDir}/{imageName}/detected_lines.png",detectedLineImage) 
        cv2.imwrite(f"{outputDir}/{imageName}/original_image.png",originalImage) 
        cv2.imwrite(f"{outputDir}/{imageName}/highlighted_image.png",highlightedImage)

    return total_lines


def remove_small_objects(img, min_size=30):

        # find all your connected components (white blobs in your image)
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
   
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        img2 = img
        # for every component in the image, you keep it only if it's above min_size

        for i in range(0, nb_components):

            if sizes[i] < min_size:
                img2[output == i + 1] = 0

        return img2


def getLines(image, threshold=20):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1)) 
    dilated = cv2.dilate(image,kernel,iterations = 5) 
    contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 


    coord = []
    for contour in contours: 
        [x,y,w,h] = cv2.boundingRect(contour)  
        coord.append((x,y,w,h)) 

    coord.sort(key=lambda tup:tup[1]) 


    i = 0
    j = 0
    lines = []
    while i < len(coord):
        _,yi,_,_ = coord[i]
        before = i
        if i+1 != len(coord):
            for j in range(i+1,len(coord)):
                _,yj,_,_ = coord[j]
                if yi+threshold < yj:
                    line = coord[i:j]
                    line = sorted(line,key=lambda tup:tup[0],reverse=True)
                    lines.append(line)
                    i = j
                    break
                else:
                    j +=1
        else:
            lines.append([coord[i]])

        if i == before:
            i+=1

    return lines



def rotate(image, angle = 10):

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),borderValue=(255,255,255))
        

        return rotated
        

def prepare_output(output):

    output = output[0,:, :,0]
    shape = output.shape
    sub_array = np.ones(shape,dtype = float) 
    img_Invert = np.abs(np.subtract(output.astype(float),sub_array))
    output = img_Invert
    output = output*255
    output = output.astype(np.uint8)

    output = np.where(output<127,255,0)
    
    output = remove_small_objects(output.astype(np.uint8))

    return output.astype(np.uint8)


def profile(image):


    proj = np.sum(image,1)
    profile = sorted(proj)[-5:]


    return np.mean(profile)


def clean_line(line):

    image = 255 - line
    proj = np.sum(image,1)


    peaks,_ = find_peaks(proj,distance = 10)

    if len(peaks)==0:
        return line
    
    i = peaks[0]
    m = 0
    for p in peaks:
        if proj[p]>m:
            i = p
            m = proj[p]

    w = peak_widths(proj, [i], rel_height=1)
    _,_, xmin, xmax = w

    xmin = int(xmin[0])
    xmax = int(xmax[0])+1
    
    return line[xmin:xmax,:]

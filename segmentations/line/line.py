from __future__ import print_function, division

from .inference import Inference_pb
from .util import line_segment, prepare_output, rotate, profile
import cv2
import numpy as np
import os

def segment(imagePath, outputDir, verbose=False):
    modelPath ="segmentations/line/models/model100_ema.pb"
    inference = Inference_pb(modelPath, imagePath, mode='L')
    predictions = inference.inference()

    imageName = imagePath.split("/")[-1].split(".")[0]
    print(imageName)

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if not os.path.exists(f"{outputDir}/{imageName}"):
        os.mkdir(f"{outputDir}/{imageName}")

    if verbose == True:
    	if not os.path.exists(f"{outputDir}/{imageName}/detected_lines"):
        	os.mkdir(f"{outputDir}/{imageName}/detected_lines")
    
    detectedLineImages = []
    
    for  output, angle in predictions:
        
        
        output = prepare_output(output)
        val = profile(output)
        detectedLineImages.append((val,output,angle))
        if verbose == True:
            cv2.imwrite(f"{outputDir}/{imageName}/detected_lines/{angle}.png",output) 
    
   
    _, bestImage, bestAngle = sorted(detectedLineImages,key=lambda tup:tup[0],reverse=True)[0]
        

    input = cv2.imread(imagePath)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    
    shape = bestImage.shape

    input = cv2.resize(input, (shape[1],shape[0]))
    input = input.astype(np.uint8)
    
    input = rotate(input, bestAngle)
        
    lines = line_segment(imageName,input,bestImage,outputDir,verbose=verbose)

    return lines, imageName, outputDir






if __name__ == '__main__':

    inputDir = "../../Dataset/Binarized/P21-Fg006-R-C01-R01-binarized.jpg"
    modelPath = "./models/model100_ema.pb"
    
    # To print intermediary images set verbose to True
    run(inputDir,'Outputs',verbose=False)

    

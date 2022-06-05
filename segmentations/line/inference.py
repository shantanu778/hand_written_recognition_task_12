from __future__ import print_function, division
import cv2
import time
from scipy.signal import argrelextrema,find_peaks,peak_widths
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from .util import load_graph

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=0.33, mode='L'):
        self.graph = load_graph(path_to_pb)
        self.path = img_list
        self.scale = scale
        self.mode = mode

    def inference(self, print_result=False, gpu_device="0"):

        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            print("Start Inference")
            timeSum = 0.0
            inputImage = self.load_img(self.path, self.scale, self.mode)
            rotatedImages = self.rotate(inputImage)
            predictions = []
            for step,image in enumerate(rotatedImages):
                aTime = time.time()
                # Run validation
                batch_x, angle = image

                if len(batch_x.shape) == 2:
                    batch_x = np.expand_dims(batch_x,2)
                batch_x = np.expand_dims(batch_x,0)

                aPred = sess.run(predictor,feed_dict={x: batch_x})

                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime
                predictions.append((aPred,angle))

                if print_result:
                    n_class = aPred.shape[3]
                    channels = batch_x.shape[3]
                    fig = plt.figure()
                    for aI in range(0, n_class+1):
                        print(aI)
                        if aI == 0:
                            a = fig.add_subplot(1, n_class+1, 1)
                            if channels == 1:
                                plt.imshow(batch_x[0, :, :, 0], cmap=plt.cm.gray)
                            else:
                                plt.imshow(batch_x[0, :, :, :])
                            a.set_title('input')
                        else:
                            a = fig.add_subplot(1, n_class+1, aI+1)
                            plt.imshow(aPred[0,:, :,aI-1], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
                            # misc.imsave('out' + str(aI) + '.jpg', aPred[0,:, :,aI-1])
                            a.set_title('Channel: ' + str(aI-1))
                    print('To go on just CLOSE the current plot.')
                    plt.savefig('output.png')
           

            print("Inference Finished!")

            return predictions



    def load_img(self, path, scale, mode):
        aImg = cv2.imread(path)
        aImg = cv2.cvtColor(aImg, cv2.COLOR_BGR2GRAY)
        sImg = cv2.resize(aImg, (int(aImg.shape[1]*scale), int(aImg.shape[0]*scale)))
        fImg = sImg


        return fImg

    def remove_connections(self, image, threshold=15000):

        inverted = 255-image

        proj = np.sum(inverted,1)
        proj = np.where(proj<threshold,0,proj)

        for row in range(image.shape[0]):
        
            if proj[row]==0:
                image[row] = 255

        return image

    def rotate(self, image, threshold = 10):

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        rotatedImages = []
        for d in range(-threshold,threshold+1):
            M = cv2.getRotationMatrix2D((cX, cY), d, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            
            rotated = self.remove_connections(rotated)
            
            rotatedImages.append((rotated,d))
        

        return rotatedImages

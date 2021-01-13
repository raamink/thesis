#!/usr/bin/python                                                                                                             

'''
author: Tobias Weis
url: http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
cited: 03-09-2020
'''

import OpenEXR
import Imath
import array
import numpy as np
import csv
import time
import datetime
import h5py
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image as pim
import os

matplotlib.use('GTK3Cairo')

def exr2depth(exr, maxvalue=1.,normalize=True):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)

    # normalize
    data = np.array(R)
    data[data > maxvalue] = maxvalue

    if normalize:
        data /= np.max(data)

    img = np.array(data).reshape(img.shape[0],-1)

    return img

def exr2flow(exr, w,h):
  file = OpenEXR.InputFile(exr)
  filePath = exr.split("/")
  fileDir = "/".join(filePath[:-1])
  fileName = filePath[-1]

  # Compute the size
  dw = file.header()['dataWindow']
  sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
  (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

  img = np.zeros((h,w,3), np.float64)
  img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
  img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)

  hsv = np.zeros((h,w,3), np.uint8)
  hsv[...,1] = 255

  mag, ang = cv2.cartToPolar(img[...,0], img[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

  return img, bgr, mag,ang

# def flow2img(img, bgr, mag, ang):



# depth_data = exr2depth("/Data/dataset/depth/depth0100.exr", maxvalue=30, normalize=False)
# flow_data  = exr2flow("/Data/dataset/flow/flow0544.exr", 1920, 1080)

# fig = plt.figure()
# # plt.imshow(depth_data)
# plt.imshow(flow_data[1])
# # plt.colorbar()
# plt.show()
# # plt.savefig("/Data/dataset/depth/depth0100.jpg")
# plt.savefig("/Data/dataset/flow/flow0544.jpg")

if __name__ == "__main__":
    dataset = "/Data/Blender"
    totalFiles = len(os.listdir(dataset+"/frames"))
    for fname in os.listdir(dataset+"/frames"):
        frameID = fname.split('.')[0]
        
        fnameDepth = f"{dataset}/depth/depth{frameID}"
        fnameFlow = f"{dataset}/flow/flow{frameID}"

        depth_data = exr2depth(fnameDepth+".exr", maxvalue=30, normalize=False)
        flow_data = exr2flow(fnameFlow+".exr", 1920, 1080)

        fig = plt.figure()
        plt.imshow(flow_data[1])
        plt.savefig(fnameFlow + ".png")
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(depth_data)
        plt.colorbar()
        plt.savefig(fnameDepth + ".png")
        plt.close(fig)

        print(f'{frameID} in list of {totalFiles} frames')

    print('ping')
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2 as cv


path = "/home/compu/ymh/FSMT/dataset/flip_semantic_re/"


for index in range(5856):
    print(index)
    
    semantic = imread(os.path.join(path, "semantic_%07d.png" % index))
    
    semantic = semantic * 255
    
    semantic = semantic.astype(np.uint8)
    
    kernel_size = 20
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    th_dil = cv.dilate(semantic,kernel,iterations=1)
    
    contours, hierarchy = cv.findContours(th_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        cv.drawContours(th_dil, [contour], -1, (255, 0, 0), -1)

    final_im = cv.erode(th_dil,kernel,iterations=1)

    blur_gaussian=True
    
    if blur_gaussian:
        final_im = cv.GaussianBlur(final_im,(15,15),0)
        _,final_im=cv.threshold(final_im, 160, 255, 0)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        final_im = cv.morphologyEx(final_im, cv.MORPH_OPEN, kernel, iterations=3)

    else:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        final_im = cv.morphologyEx(final_im, cv.MORPH_OPEN, kernel, iterations=3)
        
    final_im = np.tile(np.expand_dims(final_im,axis=2),(1,1,3))
    
    cv.imwrite("/home/compu/ymh/FSMT/dataset/flip_semantic_re/semantic_%07d.png" % index, final_im)

'''
plt.imshow(final_im, cmap = "gray")
plt.savefig("/home/compu/ymh/FSMT/dataset/flip_semantic/semantic_%d.png" % index)
plt.show()
plt.close()
'''
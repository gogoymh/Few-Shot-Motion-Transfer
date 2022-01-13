import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2 as cv


from dataset2 import pair_set

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
path2 = "/home/compu/ymh/FSMT/dataset/flip/"

a = pair_set(path1, path2, True)

#index = np.random.choice(5856, 1)
#index = 0
for index in range(5856):
    print(index)
    _, c = a.__getitem__(index)
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
       
    d = c.numpy().transpose(1,2,0)

    '''
    cv.imwrite("/home/compu/ymh/FSMT/dataset/flip_semantic/original_%d.png" % index, d)
    
    plt.imshow(d)
    plt.savefig("/home/compu/ymh/FSMT/dataset/flip_semantic/original_%d.png" % index)
    plt.show()
    plt.close()
    '''

    e1 = (c[0,:,:] >= (163/255)).float()
    e2 = (c[0,:,:] <= (173/255)).float()
    e3 = (c[1,:,:] >= (130/255)).float()
    e4 = (c[1,:,:] <= (145/255)).float()
    e5 = (c[2,:,:] >= (130/255)).float()
    e6 = (c[2,:,:] <= (145/255)).float()
    e = ((e1 + e2 + e3 + e4 + e5 + e6) == 6).float()
    
    f1 = (c[0,:,:] >= (175/255)).float()
    f2 = (c[0,:,:] <= (185/255)).float()
    f3 = (c[1,:,:] >= (145/255)).float()
    f4 = (c[1,:,:] <= (165/255)).float()
    f5 = (c[2,:,:] >= (150/255)).float()
    f6 = (c[2,:,:] <= (170/255)).float()
    f = ((f1 + f2 + f3 + f4 + f5 + f6) == 6).float()
    
    g1 = (c[0,:,:] >= (115/255)).float()
    g2 = (c[0,:,:] <= (135/255)).float()
    g3 = (c[1,:,:] >= (70/255)).float()
    g4 = (c[1,:,:] <= (95/255)).float()
    g5 = (c[2,:,:] >= (70/255)).float()
    g6 = (c[2,:,:] <= (95/255)).float()
    g = ((g1 + g2 + g3 + g4 + g5 + g6) == 6).float()
       
    j1 = (c[0,:,:] >= (190/255)).float()
    j2 = (c[0,:,:] <= (220/255)).float()
    j3 = (c[1,:,:] >= (145/255)).float()
    j4 = (c[1,:,:] <= (195/255)).float()
    j5 = (c[2,:,:] >= (145/255)).float()
    j6 = (c[2,:,:] <= (200/255)).float()
    j = ((j1 + j2 + j3 + j4 + j5 + j6) == 6).float()
        
    n1 = (c[0,:,:] >= (217/255)).float()
    n2 = (c[0,:,:] <= (240/255)).float()
    n3 = (c[1,:,:] >= (200/255)).float()
    n4 = (c[1,:,:] <= (230/255)).float()
    n5 = (c[2,:,:] >= (190/255)).float()
    n6 = (c[2,:,:] <= (230/255)).float()
    n = ((n1 + n2 + n3 + n4 + n5 + n6) == 6).float()
    
    k1 = (c[0,:,:] >= (145/255)).float()
    k2 = (c[0,:,:] <= (150/255)).float()
    k3 = (c[1,:,:] >= (90/255)).float()
    k4 = (c[1,:,:] <= (110/255)).float()
    k5 = (c[2,:,:] >= (90/255)).float()
    k6 = (c[2,:,:] <= (135/255)).float()
    k = ((k1 + k2 + k3 + k4 + k5 + k6) == 6).float()
    
    h1 = (c[0,:,:] >= (160/255)).float()
    h2 = (c[0,:,:] <= (170/255)).float()
    h3 = (c[1,:,:] >= (120/255)).float()
    h4 = (c[1,:,:] <= (140/255)).float()
    h5 = (c[2,:,:] >= (120/255)).float()
    h6 = (c[2,:,:] <= (130/255)).float()
    h = ((h1 + h2 + h3 + h4 + h5 + h6) == 6).float()
        
    boundary = j + n + f + g + k + e + h
    boundary = (boundary >= 0.5).float()
    boundary = boundary.numpy()

    #boundary[270:,:] = 0
    
    boundary = boundary * 255
    
    boundary = boundary.astype(np.uint8)
    '''
    kernel_size = 7
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    th_dil = cv.dilate(boundary,kernel,iterations=1)
    
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
    '''
    cv.imwrite("/home/compu/ymh/FSMT/dataset/flip_semantic/semantic_%07d.png" % index, boundary)

'''
plt.imshow(final_im, cmap = "gray")
plt.savefig("/home/compu/ymh/FSMT/dataset/flip_semantic/semantic_%d.png" % index)
plt.show()
plt.close()
'''
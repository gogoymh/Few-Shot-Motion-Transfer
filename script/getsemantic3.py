from skimage.io import imread, imsave
import os
import numpy as np
import cv2 as cv

def green_detect(image):
  thresh = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
          if image[i,j,0] == 34:
              if image[i,j,1] == 177:
                  if image[i,j,2] == 76:
                      thresh[i,j] = 255

  return thresh

#path2 = "/home/compu/ymh/FSMT/dataset/monalisa_semantic.png"
#path3 = "/home/compu/ymh/FSMT/dataset/monalisa_semantic2.png/"

path2 = "/data1/ymh/FSMT/dataset/monalisa_semantic.png"
path3 = "/data1/ymh/FSMT/dataset/monalisa_semantic2.png"

#names = os.listdir(path2)

for i in range(1):
    print(i)
    #name = os.path.join(path2, names[i])
    img = imread(path2)
    thresh = green_detect(img)
    
    kernel_size=10
    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    th_dil=cv.dilate(thresh,kernel,iterations=1)

    contours, hierarchy = cv.findContours(th_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv.drawContours(th_dil, [contour], -1, (255, 0, 0), -1)

    final_im=cv.erode(th_dil,kernel,iterations=1)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    final_im = cv.morphologyEx(final_im, cv.MORPH_OPEN, kernel, iterations=3)
  
    final_im=np.tile(np.expand_dims(final_im,axis=2),(1,1,3))
    
    #cv.imwrite(os.path.join(path3, names[i]), final_im)
    cv.imwrite(path3, final_im)
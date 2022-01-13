import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= '/home/compu/ymh/FSMT/dataset/tmp/'
pathOut = '/home/compu/ymh/FSMT/dataset/tmp.mp4'
fps = 29.97

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#print(files[:10])

files.sort()
#print(files[:10])


for i in range(len(files)):
    print(i)
    filename = pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'FMP4'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()

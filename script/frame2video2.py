import cv2
import numpy as np
import os
from os.path import isfile, join
from torchvision import transforms

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor()
                ]) 


org_path = "/home/compu/ymh/FSMT/dataset/yerin_frame/"
joint_path = "/home/compu/ymh/FSMT/dataset/yerin_rendered/"

pathIn = '/home/compu/ymh/FSMT/save/inference66/'
pathOut = '/home/compu/ymh/FSMT/dataset/inference66.mp4'
'''

org_path = "/data1/ymh/FSMT/dataset/yerin_frame/"
joint_path = "/data1/ymh/FSMT/dataset/yerin_rendered/"

pathIn = "/data1/ymh/FSMT/save/inference61/"
pathOut = "/data1/ymh/FSMT/dataset/inference61.mp4"
'''
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
    
    org = cv2.imread(os.path.join(org_path, "frame_%07d.png" % i))
    joint = cv2.imread(os.path.join(joint_path, "frame_%07d_rendered.png" % i))
    
    org = change(org)
    org = org.numpy().transpose(1,2,0)
    org = org * 255
    org = org.astype('uint8')
    
    joint = change(joint)
    joint = joint.numpy().transpose(1,2,0)
    joint = joint * 255
    joint = joint.astype('uint8')
    
    frame = np.concatenate((org, joint, img), axis=1)
    
    frame_array.append(frame)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'FMP4'), fps, (576,320))

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()

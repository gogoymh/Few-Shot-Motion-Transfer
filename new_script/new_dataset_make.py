from skimage.io import imread, imsave
import os
from skimage.color import gray2rgb


joint_path = "/home/compu/ymh/FSMT/dataset/data/joint_tmp/"
img_path = "/home/compu/ymh/FSMT/dataset/data/style_tmp/"
save_path = "/home/compu/ymh/FSMT/dataset/data/new_style/"

joint_names = os.listdir(joint_path)
length = len(joint_names)


for index in range(length):
    print(index, end=" ")
    save = True
    joint_name = joint_names[index]
        
    img_name = joint_name.replace("_rendered", "")
    try:
        img = imread(os.path.join(img_path, img_name))
    except:
        img_name = img_name.replace("png", "jpg")
        img = imread(os.path.join(img_path, img_name))
        
    if len(img.shape) == 2:
        img = gray2rgb(img)
    
    if img.shape[0] == 1024:
        if img.shape[1] == 1024:
            save = False
            print("No")
            continue
            
    if min(img.shape[0], img.shape[1]) < 100:
        save = False
        print("No")
        continue
    
    if save:
        imsave(os.path.join(save_path, img_name), img)
        print("Yes")

from skimage.io import imread, imsave
import os
#import matplotlib.pyplot as plt

#input_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//LIP_input//"

#input_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//smp//yerin//"

#input_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//MHPv2_input//"
input_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//Face_input//"
images = os.listdir(input_path)

input_with_img_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//Face_with_joint//"
output_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//Face//"

#output_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//LIP_done//"
#output_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//MHPv2_output//"

#input_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//input//"
#output_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//output//"

#input_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//smp//yerin_save//"

input_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_input//"
input_with_img_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_input_with_img//"
output_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_output//"

idx = 75650
for i in range(5695):
    print(i, idx)
    input_img_name = images[i]
    input_with_img_name = images[i]
    output_img_name = images[i].replace("_rendered", "")
        
    input_img = imread(os.path.join(input_path, input_img_name))
    input_with_img = imread(os.path.join(input_with_img_path, input_img_name))
    output_img = imread(os.path.join(output_path, output_img_name))
    
    input_save_name = "%07d.png" % idx
    input_with_img_save_name = "%07d_rendered.png" % idx
    output_save_name = "%07d.png" % idx

    imsave(os.path.join(input_save, input_save_name), input_img)
    imsave(os.path.join(input_with_img_save, input_with_img_save_name), input_with_img)
    imsave(os.path.join(output_save, output_save_name), output_img)
    
    idx += 1

    
    
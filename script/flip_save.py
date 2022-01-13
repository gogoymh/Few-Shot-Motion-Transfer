from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

input_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//yerin_rendered//"
save_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//tmp_rendered//"

save_idx = 0

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.CenterCrop((720,432)),
                transforms.ToTensor()
                ]) 

for index in range(5940):
    print(index)
    img = imread(os.path.join(input_path, "frame_%07d_rendered.png" % index))
    #flip_img = np.fliplr(img)
    
    img = change(img)
    img = img.numpy().transpose(1,2,0)
    img = img * 255
    img = img.astype("uint8")
    imsave(os.path.join(save_path, "frame_%07d_rendered.png" % index), img)
    
    #imsave(os.path.join(save_path, "%07d.png" % (save_idx+1)), flip_img)

    #save_idx += 1
    
#plt.imshow(img)
#plt.show()
#plt.close()
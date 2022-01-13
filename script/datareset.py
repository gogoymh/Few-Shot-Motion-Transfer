from skimage.io import imread, imsave
import os

'''
a = [8460, 8462, 8470, 8474, 8476, 8482, 8494, 8496, 8506, 8508, 8510, 8512, 8518, 8568, 8570, 8571, 8574, 8575, 8576, 8577, 8578, 8579, 8580,
     8581, 8582, 8583, 8584, 8585, 13320, 13322, 13325, 13337, 13339, 13347, 13365, 13367, 14994, 14998, 15088, 15089, 15276, 15277, 15283, 15285,
     15287, 15724, 15725, 17368, 18602, 18603, 18854, 18862, 21930, 21932, 22927, 25038, 25039, 25083, 26298, 26300, 26334, 27522, 28697, 29613]
a = a + list(range(8000,8005)) + list(range(11150,11170)) + list(range(11949,11966)) + list(range(13276,13280)) + list(range(13330,13335))
a = a + list(range(13349,13362)) + list(range(15002,15016)) + list(range(15091,15098)) + list(range(15129,15134)) + list(range(15500,15528))
a = a + list(range(15832,15836)) + list(range(15838,15860)) + list(range(15867,15874)) + list(range(16731,16742)) + list(range(17258,17282))
a = a + list(range(20284,20327)) + list(range(21944,21948)) + list(range(25260,25297)) + list(range(25781,25877)) + list(range(26385,26471))
a = a + list(range(26608,26630)) + list(range(26721,26758)) + list(range(27194,27232)) + list(range(27722,27741)) + list(range(27779,27790))
a = a + list(range(28714,28717)) + list(range(29478,29572)) + list(range(29585,29598)) + list(range(33015,33064)) + list(range(34082,34095))
'''
a = list(range(5414,5430))

a = sorted(a)

reset_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
reset_rendered_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_rendered//"

reset_save_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//reset1//"
reset_rendered_save_path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//reset1_rendered//"

idx = 5856
for i in range(16):
    print(i, idx)
    img_name = "%07d.png" % idx
    rendered_name = "%07d_rendered.png" % idx
    
    img = imread(os.path.join(reset_path, img_name))
    rendered = imread(os.path.join(reset_rendered_path, rendered_name))
    
    img_save_name = "%07d.png" % a[i]
    rendered_save_name = "%07d_rendered.png" % a[i]
    
    imsave(os.path.join(reset_save_path, img_save_name), img)
    imsave(os.path.join(reset_rendered_save_path, rendered_save_name), rendered)
    
    idx += 1
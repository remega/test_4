import cv2
import numpy as np

Image = np.zeros((360,180))

for x in range(360):
    for y in range(60):
        # print("x,y",x,y) # 359,179
        Image[x][y] = 0.0

for x in range(360):
    for y in range(60,120):
    #   print("x1,y1",x,y) # 359,179
      Image[x][y] = 1.0

for x in range(360):
    for y in range(120,180):
      print("x1,y1",x,y) # 359,179
      Image[x][y] = 3.0

print("np.shape(Image)",np.shape(Image))

path = '/home/minglang/test_code/'
name = 't1'
env_id = 'A1'
Image = np.transpose(Image)
heatmap = Image
file_name = path+env_id+'_'+name+'.jpg'
heatmap_width = 360
heatmap_height = 180

cv2.imwrite(path+env_id+'_'+name+'.jpg',heatmap)

# Image_read0 = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Image_read1 = cv2.resize(Image_read0,(heatmap_width, heatmap_height))
# print("np.shape(Image_read),np.max(Image_read): ",np.shape(Image_read1),np.max(Image_read1))

# for x in range(180):
#     for y in range(360):
#     #   print("x1,y1",x,y) # 359,179
#       if(Image_read[x][y] == 255):
#           print("x,y:", x,y)

for step in range(90):
    file_name1 = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/'+'KingKong'+'_'+str(step)+'.jpg'
    Image_read = cv2.imread(file_name1, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    print("np.shape(Image_read): ",np.shape(Image_read))
    Image_read1 = cv2.resize(Image_read,(heatmap_width, heatmap_height))
    print("np.max(Image_read1): ", np.max(Image_read1))

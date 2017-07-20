import cv2
import numpy as np

i  = 0
Image = np.zeros((180,360))

for x in range(180):
    for y in range(360):
        # print("x,y",x,y) # 359,179
        Image[x][y] = 0.0

for x in range(1):
    for y in range(345,355):
    #   print("x1,y1",x,y) # 359,179
      Image[x][y] = 1.0

for x in range(1):
    for y in range(355,360):
    #   print("x1,y1",x,y) # 359,179
      Image[x][y] = 2.0

print("np.shape(Image)",np.shape(Image))


path = '/home/minglang/test_code/'
name = 't1'
env_id = 'A1'
heatmap = Image
file_name = path+env_id+'_'+name+'.jpg'
heatmap_width = 360
heatmap_height = 180
# write the data
heatmap_1 = heatmap*(1.0/np.max(heatmap))*255.0

cv2.imwrite(path+env_id+'_'+name+'.jpg',heatmap_1,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
"print the raw value"
for x in range(180):
    for y in range(360):
        if heatmap_1[x][y] > 0:
            # i += Image_read0[x][y]
            i = i + 1
            print(">>>>>>>>>>>>>>>>i,x,y,heatmap_1[x][y]: ",i,x,y,heatmap_1[x][y])

# read data
Image_read0 = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
Image_read00 = cv2.resize(Image_read0,(heatmap_width, heatmap_height))
Image_read1 = Image_read00/255.0
print("np.shape(Image_read0),np.max(Image_read0): ",np.shape(Image_read0),np.max(Image_read0))

"print the read value"
for x in range(180):
    for y in range(360):
        if Image_read0[x][y] > 0:
            # i += Image_read0[x][y]
            i = i + 1
            print("i,x,y,Image_read00[x][y],Image_read1[x][y]: ",i,x,y,Image_read00[x][y],Image_read1[x][y])

"display the read data"
# for x in range(180):
#     for y in range(360):
#     #   print("x1,y1",x,y) # 359,179
#       if(Image_read[x][y] == 255):
#           print("x,y:", x,y)

"read the envs_li.py's heatmap"
# for step in range(90):
#     file_name1 = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/'+'KingKong'+'_'+str(step)+'.jpg'
#     Image_read = cv2.imread(file_name1, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#     print("np.shape(Image_read): ",np.shape(Image_read))
#     Image_read1 = cv2.resize(Image_read,(heatmap_width, heatmap_height))
#     print("np.max(Image_read1): ", np.max(Image_read1))

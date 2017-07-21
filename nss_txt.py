#coding=utf-8
# import cv2
import numpy as np


class for_nss():

    def __init__(self):
        self.heatmap_width = 360
        self.heatmap_height = 180
        pass

    def data_test(self):
        i  = 0
        Image = self.test_array_1(select = 0)

        path = '/home/minglang/test_code/'
        name = 't1'
        env_id = 'A1'
        heatmap = Image
        file_name = path+env_id+'_'+name+'.jpg'

        # write the data
        self.save_groundtruth_txt_for_nss(heatmap,path)

        "print the raw value"
        for x in range(self.heatmap_height):
            for y in range(self.heatmap_width):
                if heatmap[x][y] > 0:
                    pass
                    # i += Image_read0[x][y]
                    # i = i + 1
                    print(">>>>>>>>>>>>>>>>i,x,y,heatmap_1[x][y],heatmap.dtype: ",i,x,y,heatmap[x][y],heatmap.dtype)

        # read data
        data = self.read_groundtruth_txt_for_nss(path = path)
        print(">>>>>>>>>>>>>>>>>>>>data,np.shape(data): ", data.astype('float'),np.shape(data))
        for x in range(self.heatmap_height):
            for y in range(self.heatmap_width):
                if data[x][y].astype('float') >= 1:
                    print(">>>>>>>>>>>>>>>>i,x,y,heatmap_1[x][y]: ",x,y,data[x][y])

        Image_read0 = 0
        Image_read00 = 0
        Image_read1 = 0

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
    def read_groundtruth_txt_for_nss(self,path):
        x = []
        y = []
        value = []

        f = open(path + 'groudtruth_for_nss.txt',"r")
        lines = f.readlines()#读取全部内容
        # slip the value
        for line in lines:
            x0, y0,value0 = line.split()
            x.append(x0)
            y.append(y0)
            value.append(value0)
        value = np.array(value)
        value = np.resize(value,(self.heatmap_height, self.heatmap_width))
        print("np.shape(value): ", np.shape(value))

        return value

    "save groundtruth data as txt"
    def save_groundtruth_txt_for_nss(self,value,path):
        f = open(path+'groudtruth_for_nss.txt','w') # 'w' mode will clear the formore data
        for x in range(self.heatmap_height):
            for y in range(self.heatmap_width):
                print_string = '\t'
                # print_string += 'step' + '\t'
                print_string += str(x) + '\t'+str(y) +'\t'
                # print_string += 'ave_cc' + '\t'
                print_string += str(value[x][y]) + '\t'
                print_string += '\n'
                f.write(print_string)
        f.close()


    def test_array_1(self,select):
        if select == 0:
            Image = np.zeros((180,360))
            for x in range(180):
                for y in range(360):
                    # print("x,y",x,y) # 359,179
                    Image[x][y] = 0.0

            for x in range(1):
                for y in range(0,5):
                #   print("x1,y1",x,y) # 359,179
                  Image[x][y] = 1.0

            for x in range(1):
                for y in range(5,10):
                #   print("x1,y1",x,y) # 359,179
                  Image[x][y] = 2.0

        # if select == 1:
        #     i1 = 0
        #     Image = []
        #     for x in range(2):
        #         for y in range(3):
        #             # print("x,y",x,y) # 359,179
        #             i1 += 1
        #             Image[x][y] = i1

        print("np.shape(Image)",np.shape(Image))

        return Image


if __name__ == '__main__':
    test1 = for_nss()
    test1.data_test()

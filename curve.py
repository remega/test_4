#coding=utf-8
import matplotlib.pyplot as plt
import cv2
import numpy as np

class curve():
    def __init__(self):
        i  = 0
        Image = np.zeros((180,360))

    #创建绘图对象，figsize参数可以指定绘图对象的宽度和高度，单位为英寸，一英寸=80px
    # plt.figure(figsize=(8,4))
    # plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2,alpha = 0.5)
    #
    # #X轴的文字
    # plt.xlabel("Time(s)")
    #
    # #Y轴的文字
    # plt.ylabel("Volt")
    #
    # #图表的标题
    # plt.title("PyPlot First Example")
    #
    # #Y轴的范围
    # plt.ylim(-1.2,1.2)
    #
    # #显示图示
    # plt.legend()
    #
    # #show picture
    # plt.show()
    #
    # #保存图
    # plt.savefig("sinx.jpg")

    def plot_curve(self,x_data,y_data,figure_size,alpha,save_path,y_label,y_lable_range):
        plt.figure(figsize = figure_size)
        plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2,alpha = 0.5)

        #X轴的文字
        plt.xlabel("Frame")
        #Y轴的文字
        plt.ylabel(y_label)
        #Y轴的范围
        plt.ylim((-y_lable_range,y_lable_range))
        #显示图示
        # plt.legend()
        axes = plt.subplot(111)
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.spines['bottom'].set_color('none')
        axes.spines['left'].set_color('none')
        #保存图
        file_name = save_path + "a.jpg"
        plt.savefig(file_name)
        # shouw picture
        plt.show()



if __name__ == '__main__':
    x = x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    a = curve()
    a.plot_curve(
               x_data = x,
               y_data = y,
               figure_size = (8,4),
               alpha = 0.9,
               save_path = "/home/minglang/test_code/",
               y_label = 'Longitude',
               y_lable_range = 1
            )

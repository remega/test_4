#coding=utf-8
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

cc = []
fcb_sigma = []
weight_prediction =[]

path_source = '/home/minglang/PAMI/cc_result/ours_and_ground/all_videos_ave_cc_with_fcb_fitting/all_for_3d_fitting_ave_cc_0902.txt'
f = open(path_source, "r")
lines = f.readlines() #read all lines
print(lines)
for line in lines:
    line = line.split()
    print(str(line[0]) + '_' + str(line[1])+ '_' + str(line[2]))
    cc.append(float(line[0]))
    fcb_sigma.append(float(line[1]))
    weight_prediction.append(float(line[2]))

print('>>>>>>>>>>>>>np.max(cc): ', np.max(cc))
print('>>>>>>>>>>>>>>cc.index(max(cc)): ',cc.index(max(cc)))
print(">>>>>>>>>>>>>>>>>>>>: fcb,weight: ",fcb_sigma[cc.index(max(cc))], weight_prediction[cc.index(max(cc))])


'scatter figure'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = fcb_sigma
Y = weight_prediction
Z = cc
ax.scatter(X, Y, Z)
# plt.show()




#coding:utf-8


'meshgrid'
# def f(x,y):
#     z = (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
#     return z
#
# n = 256
#
# #均匀生成-3到3之间的n个值
# x = np.linspace(-3,3,n)
# y = np.linspace(-3,3,n)
# #生成网格数据
# X,Y = np.meshgrid(x,y)
# Z = f(x,y)
#
# fig = plt.figure()
#
# #第四个子图，第二行的第二列
# subfig4 = fig.add_subplot(1,1,1,projection='3d')
# #画三维图
# # surf4 = subfig4.plot_surface(X, Y, f(X,Y), rstride=1, cstride=1, cmap='jet',
# #         linewidth=0, antialiased=False)
# subfig4.scatter(x, x, Z)
# #设置色标
# # fig.colorbar(surf4)
# #设置标题
# plt.title('plot_surface+colorbar')
#
# plt.show()

'detailed'
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
#
# X = fcb_sigma
# Y = weight_prediction
# Z = cc
#
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
#
# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)
#
# plt.show()

from constant import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import scipy.special as scp
from pathlib import Path
import imageio
import os
import os.path

class IGRF:                              # 定义类IGRF
    def readdata(self, filename):        # 定义类中的方法
    # 读取数据

        G = []
        n = []
        m = []
        data = np.zeros((195, 26))      # 建立一个形状为195×26的零矩阵
        with open(filename) as f:       # 打开名为filename的文件并在该程序中称其为f
            lines = f.readlines()       # 按行读取文件中的内容
            i = 0
            for line in lines:
                lineData = line.strip().split() 
                # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                # split() 通过指定分隔符对字符串进行切片
                G.append(lineData[0])        # 在G列表的末尾加上lineData的第一个元素
                n.append(int(lineData[1]))   # 在n列表的末尾加上lineData的第二个元素并将其强制转换为整数类型
                m.append(int(lineData[2]))   # 在m列表的末尾加上lineData的第三个元素并将其强制转换为整数类型
                data[i,:] = lineData[3:]     # 取第i行的全部，将lineData的第四个元素开始赋给data
                i = i + 1
        g = np.zeros(N1)                     # 建立一个形状为1×N1的零矩阵
        for i in range(N1):
            g[i] = 0 if G[i] == 'g' else np.pi/2 # 如果G[i]为字符g则g[i]为0，否则g[i]为π/2

        return g, n, m, data

    def V(self, g, n, m, data):
        # 计算非偶极子场
        ans = np.zeros(shapex)      # 建立一个[100, 200, 26]的零矩阵
        for i in range(N2):
            for j in range(N1):
                if n[j] == 1:
                    # 去掉偶极子场
                    continue
                e = 1 if m[j] == 0 else 2   # 如果m[j]=0则e=1，否则e=2
                ans[:,:,i] = ans[:,:,i] - (-1)**(m[j])*(n[j]+1) * data[j,i]*np.cos(m[j]*Phi-g[j]) * \
                    (e * factorial(n[j]-m[j]) / factorial(n[j]+m[j]))**0.5 * \
                        (scp.lpmv(m[j], n[j], np.cos(Theta)))
                # 依次将计算得到的磁场强度值填充到每一个位置上
    
        ans.tofile('data.dat', sep = ' ', format = '%f')
        # tofile()将数组中的数据以二进制格式写进文件，不保存数组形状和元素类型等信息

    def drawpicture(self, path, save = False):
        # 画图

        plt.ion()  
        # 打开交互模式
        # 读入生成的数据，并将读入的数据改为[100, 200, 26]的形状
        result = np.fromfile('data.dat', dtype = float, sep = ' ').reshape(shapex)
        # 画布大小，定义其为10cm×7cm
        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_axes([0.1,0.1,0.85,0.85])
        # 从画布 10% 的位置开始绘制, 宽高是画布的 85%。

        for index in range(N2):
            plt.cla()
            # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
            plt.title('IGRF--'+str(1900+index*5))
            # 绘制地图（世界地图）
            map = Basemap(ax = ax1)
            # 建立一个地图
            map.drawcoastlines()
            # 绘制海岸线
            map.drawparallels(np.arange(-90,90,20),labels=[1,0,0,1])
            # 绘制纬线，labels设置在图片的上下左右标出刻度：[左，上，右，下]，0就是不要标，1就是标上；fontsize设置字号的大小。
            map.drawmeridians(np.arange(-180,180,30),labels=[1,0,0,1])
            # 绘制经线
            X,Y = map(Phi, Theta)
            map.contour(X*t, 90 - Y*t, result[:,:,index], 15)
            # t = 180 / np.pi，Phi代表经度，90-Theta代表纬度,15为等值线的条数

            # 将每年的非偶极子场的图保存
            if save:
                filename = 'IGRF--'+str(1900+index*5)+'.png'
                plt.savefig(path+filename)

            plt.pause(0.1)

        plt.ioff()
        plt.show()


    def creategif(self, path, gif_name):
        # 将png图片保存为gif动图

        frames = []

        pngFiles = os.listdir(path)
        #os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表
        image_list = [os.path.join(path, f) for f in pngFiles]
        #os.path.join()函数用于路径拼接文件路径，可以传入多个路径
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 将图片都读取后连接到一起
        imageio.mimsave(gif_name, frames, 'GIF', duration = 0.3)
        #生成动图，duration为每一帧停留的时间


if __name__ == '__main__':
    g, n, m, data = IGRF().readdata('D:\Github\IGRF13\igrf13coeffs.txt')
    file = Path('data.dat')
    if not file.is_file():
        # 计算一次，避免重复计算
        IGRF().V(g, n, m, data)

    path = 'D:\Github\IGRF13\pngfile\ '
    #注意要在反斜杠\后加一个空格
    IGRF().drawpicture(path, save=True)
    IGRF().creategif('D:\Github\IGRF13\pngfile', 'IGRF.gif')


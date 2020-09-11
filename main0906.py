#在study09053.py的程序代码上继续进行测试

'''
依据xls文件找到对应的dicom文件并将之转化为jpg格式的程序

vesion：20200906
author：Quinlan

注意事项：读取的dicom文件为int16，参考并编写了转为uint8的代码，具体效果仍需后面验证，所以0906暂时使用这个版本
'''


import xlrd
import os
import pydicom
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np


#xls=os.chdir()   #切换路径
path=r'E:\Data\LIDC\Dataframe\\reshaixuan0101'
xlspath=os.listdir(path)
pathdcm=r'H:\公开数据库\LIDC\LIDC-IDRI'
tmppathdcm='LIDC-IDRI-'

outpath=r'E:\Data\LIDC\jpgdata09072'
jpgname=0


#for i in range(0,len(xlspath)):
for i in range(len(xlspath)):
    xls_spilt=os.path.splitext(xlspath[i])[0]
    #reshaixuancase1allnodle.xls    是这个格式的文件名，如果文件名不同还需要再进行修改
    xls_spilt_num=xls_spilt[14:-8]
    xls=os.path.join(path,xlspath[i])
    print(xls)
    xls_data = xlrd.open_workbook(xls)
    table = xls_data.sheet_by_index(0)
    dcmnumber = table.col_values(0)
    dcmnumber = [int(x) for x in dcmnumber]
    dcmnumber = np.unique(dcmnumber)
    print(dcmnumber)
    #print(xls_spilt_num)


    #将xls文件和队应的原始数据集联系起来
    casenum=str(xls_spilt_num).zfill(4)
    casedcm=tmppathdcm+casenum

    casepath=os.path.join(pathdcm,casedcm)


    #得到所有dicom文件名，进入文件夹的下两级
    # dcmway_1=os.listdir(r"%s" % casepath)
    # dcmway_2=os.path.join(casepath,dcmway_1[0])
    dcmway_2 = os.path.join(casepath, 'Dicom')
    dcmway_3=os.listdir(r"%s" % dcmway_2)
    dcmway=os.path.join(dcmway_2,dcmway_3[0])
    #print(dcmway)
    #dcmname=glob.glob(r"%s\*.dcm" % dcmway)



    #根据xls文件中的dicom编号去找到对应的dicom文件
    for j in dcmnumber:

        dcm_name='%s.dcm' % j
        dcmpath=os.path.join(dcmway,dcm_name)
        #print(dcmpath)
        ds=pydicom.read_file(r"%s" % dcmpath)
        img = ds.pixel_array
        # plt.imshow(img, "gray")
        # plt.show()
        h = np.max(img)
        l = np.min(img)
        lungwin = np.array([l * 1., h * 1.])
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg = (newimg * 255).astype('uint8')



        jpgname+=1
        print(jpgname)
        jpgname_fill=str(jpgname).zfill(4)




        #imageio.imwrite("%s/%s.jpg" % (outpath, jpgname_fill), img)
        imageio.imwrite("%s/%s.jpg" % (outpath, jpgname_fill), newimg)



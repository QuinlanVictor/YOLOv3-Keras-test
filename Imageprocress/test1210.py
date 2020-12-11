'''
根据xls文件找到dicom文件，进行图像预处理

代码来源 E:\Files\Repositories\kerasYolov4\test\imgprocess

version 20201210
author  Quinlan
'''

import os
import SimpleITK as sitk
import scipy.ndimage as ndimage

import xlrd
import os
import pydicom
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np


'''窗位窗宽的调整'''
def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
   newImg = sitk.GetImageFromArray(savedImg)
   newImg.SetOrigin(origin)
   newImg.SetDirection(direction)
   newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
   sitk.WriteImage(newImg, saved_name)

'''进行预处理的测试'''
#xls=os.chdir()   #切换路径
path=r'E:\Data\LIDC\Dataframe\\reshaixuan0101'
xlspath=os.listdir(path)
pathdcm=r'G:\Dataachive\LIDC\LIDC-IDRI'
tmppathdcm='LIDC-IDRI-'

outpath=r'E:\Data\LIDC\dicom1210'



#for i in range(2):
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
        ct = sitk.ReadImage(dcmpath)
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        xyz_thickness = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        config = {
            'xyz_thickness': [1.0, 1.0, 1.0]
        }
        # step1 重采样
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
                                           ct.GetSpacing()[0] / config['xyz_thickness'][0],
                                           ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=3)

        # step2 窗位窗宽
        tran = window_transform(ct_array, 1500, -400, normal=False)

        saved_case = os.path.join(outpath,casedcm)

        if not os.path.exists(saved_case):
            os.mkdir(saved_case)

        saved_name = os.path.join(outpath, casedcm, dcm_name)

        saved_preprocessed(tran, origin, direction, xyz_thickness, saved_name)

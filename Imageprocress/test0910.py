'''

进行8位向24位的转换

vesion：20200910
author：Quinlan


'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import shutil

path='E:\Data\LIDC\Segmentationjpg/09083'
newpath='E:\Data\LIDC\Segmentationjpg/0910'

#fileList = []

files = os.listdir(path)
i=0
for f in files:

   imgpath = path + '/' + f
   img=Image.open(imgpath).convert('RGB')
   dst=os.path.join(newpath,f)
   img.save(dst)

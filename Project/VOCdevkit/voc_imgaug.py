# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Augmentation

# %%
import os
#os.chdir(r'D:/data_augmentation')


# %%
'''
usage:
python voc_imgaug.py -bd VOC2007 -sp VOC2007-imgaug
'''

import argparse
parser = argparse.ArgumentParser()
'''
Command line options
'''
parser.add_argument(
    '-bd','--base_dataset', type=str,
    help='path to the dataset to be solved',default = 'VOC2007_normal')

parser.add_argument(
    '-sp','--save_path', type=str,
    help='path to save the processing results', default = 'new')

options = parser.parse_args()
print(options)

# %%
# 对VOC2007数据进行拷贝到new
import shutil #拷贝模块
src_path = options.base_dataset
res_path = options.save_path
# os.makedirs(res_path, exist_ok=True)
if os.path.exists(res_path):
    shutil.rmtree(res_path)
shutil.copytree(src_path, res_path)


# %%
annotation_path = os.path.join(res_path,'Annotations')
image_path = os.path.join(res_path,'JPEGImages')
imageset_path = os.path.join(res_path,'ImageSets')

# %% [markdown]
# # xml和jpg的match检查

# %%
import glob
im_paths = glob.glob(os.path.join(image_path,'*.jpg'))
for i in im_paths:
    xml_p = i.replace('JPEGImages','Annotations')
    xml_p = xml_p.replace('.jpg','.xml')
    if not os.path.exists(xml_p):
        print(i)
        os.remove(i)
xm_paths = glob.glob(os.path.join(annotation_path,'*.xml'))
for i in xm_paths:
    im_p = i.replace('Annotations','JPEGImages')
    im_p = im_p.replace('.xml','.jpg')
    if not os.path.exists(im_p):
        print(i)
        os.remove(i)
    
    

# %% [markdown]
# # 翻转

# %%
from PIL import Image
from xml.etree import ElementTree as ET

def flip(img_path):
    im = Image.open(img_path)

    im_FH = im.transpose(Image.FLIP_LEFT_RIGHT)
    
    im_FH.save(img_path.replace('.jpg','_FH.jpg')) # transpose and save
    im_FV = im.transpose(Image.FLIP_TOP_BOTTOM)
    
    im_FV.save(img_path.replace('.jpg','_FV.jpg'))


    # for calculating new bbox
    w, h = im.size
    #print('width = %s' %(w),'height = %s' %(h))

    # extract all bndboxes in the according xml file
    
    xml_file = img_path.replace('JPEGImages','Annotations')
    xml_file = xml_file.replace('.jpg','.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 修改xml的filename
    root[1].text = root[1].text.replace('.jpg','_FH.jpg')

    for member in root.findall('object'):
        # do horizontal flip
        xmin = member[4][0].text
        xmax = member[4][2].text
        ymin = member[4][1].text
        ymax = member[4][3].text

        member[4][0].text = str(w - int(xmax))
        member[4][2].text = str(w - int(xmin))

    # FH represents flip horizontal
    tree.write(xml_file.replace('.xml','_FH.xml'), xml_declaration=True, encoding="utf-8")

    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 修改xml的filename
    root[1].text = root[1].text.replace('.jpg','_FV.jpg')

    for member in root.findall('object'):
        # do vertical flip
        xmin = member[4][0].text
        xmax = member[4][2].text
        ymin = member[4][1].text
        ymax = member[4][3].text

        member[4][1].text = str(h - int(ymax))
        member[4][3].text = str(h - int(ymin))

    # FH represents flip horizontal
    tree.write(xml_file.replace('.xml','_FV.xml'), xml_declaration=True, encoding="utf-8")


# %%
import glob
im_paths = glob.glob(os.path.join(image_path,'*.jpg'))
for im_path in im_paths:
    print('flipping %s ....'%im_path, end='\r')
    try:
        flip(im_path)
    except:
        print('Error occur in:', im_path)
        pass
    

# %% [markdown]
# # 旋转

# %%
import sys
def img_fill(n_w, n_h, im_w, im_h, img):
    if (n_w<im_h) or (n_h<im_h):
        print('Error! the new image must be bigger than raw image!')
        sys.exit()
        
    p = Image.new(img.mode, (n_w, n_h), (255, 255, 255))
    offset1 = int((n_w - im_w)/2)
    offset2 = int((n_h - im_h)/2)
    p.paste(img, (offset1, offset2, im_w+offset1, im_h+offset2), None)
    return p


# %%
def img_rotate(img_path):
    im = Image.open(img_path)
    w,h = im.size

    # before rotate, we need to fill the im into a square img
    # im_R90 = img_fill(max(w,h), max(w,h), w, h, im)
    im_R90 = im.rotate(90,expand=True)
    # crop the blank area
    # im_R90 = im_R90.crop((int((w-h)/2),0,int(w-(w-h)/2),w))
    im_R90.save(img_path.replace('.jpg','_R90.jpg'))

    im_R180 = im.transpose(Image.ROTATE_180)
    im_R180.save(img_path.replace('.jpg','_R180.jpg'))
    im_R270 = im_R90.transpose(Image.ROTATE_180)
    im_R270.save(img_path.replace('.jpg','_R270.jpg'))

    xml_file = img_path.replace('JPEGImages','Annotations')
    xml_file = xml_file.replace('.jpg','.xml')

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 修改xml的filename
    root[1].text = root[1].text.replace('.jpg','_R90.jpg')

    for member in root.findall('object'):
        # do rotate 90
        xmin = int(member[4][0].text)
        xmax = int(member[4][2].text)
        ymin = int(member[4][1].text)
        ymax = int(member[4][3].text)

        xmin1 = ymin
        ymin1 = w - xmin
        xmax1 = ymax
        ymax1 = w - xmax

        xmin = xmin1
        ymin = ymax1
        xmax = xmax1
        ymax = ymin1

        member[4][0].text = str(xmin)
        member[4][1].text = str(ymin)
        member[4][2].text = str(xmax)
        member[4][3].text = str(ymax)        

    root.find('size')[0].text = str(h)
    root.find('size')[1].text = str(w)

    # R90 represents rotate 90 degrees
    tree.write(xml_file.replace('.xml','_R90.xml'), xml_declaration=True, encoding="utf-8")


    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 修改xml的filename
    root[1].text = root[1].text.replace('.jpg','_R180.jpg')

    for member in root.findall('object'):
        # do rotate 180
        xmin = int(member[4][0].text)
        xmax = int(member[4][2].text)
        ymin = int(member[4][1].text)
        ymax = int(member[4][3].text)

        xmin1 = w - xmax
        xmax1 = w - xmin
        ymin1 = h - ymax
        ymax1 = h - ymin

        xmin = xmin1
        xmax = xmax1
        ymin = ymin1
        ymax = ymax1

        member[4][0].text = str(xmin)
        member[4][1].text = str(ymin)
        member[4][2].text = str(xmax)
        member[4][3].text = str(ymax)

    # R180 represents rotate 180 degrees
    tree.write(xml_file.replace('.xml','_R180.xml'), xml_declaration=True, encoding="utf-8")

    # when we rewrite R270, wo can use the R90 file
    tree = ET.parse(xml_file.replace('.xml', '_R90.xml'))
    root = tree.getroot()

    # 修改xml的filename
    root[1].text = root[1].text.replace('_R90.jpg','_R270.jpg')

    for member in root.findall('object'):
        # do rotate 270
        xmin = h - int(member[4][2].text)
        xmax = h - int(member[4][0].text)
        ymin = w - int(member[4][3].text)
        ymax = w - int(member[4][1].text)

        member[4][0].text = str(xmin)
        member[4][1].text = str(ymin)
        member[4][2].text = str(xmax)
        member[4][3].text = str(ymax)

    root.find('size')[0].text = str(h)
    root.find('size')[1].text = str(w)
    # R270 represents rotate 270 degrees
    tree.write(xml_file.replace('.xml','_R270.xml'), xml_declaration=True, encoding="utf-8")


# %%
for im_path in im_paths:
    print('rotating %s ....'%im_path, end='\r')
    try:
        img_rotate(im_path)
    except:
        print('Error occur in:', im_path)
        pass

# %% [markdown]
# # 增强图像的对比度

# %%
# image preprocess

import copy
import numpy as np
import cv2
import argparse, textwrap


# 高斯滤波
def gaussian_filter(image):
    dst = cv2.GaussianBlur(image, (3, 3), 0)  
    return dst

# 对比度增强2
def contrast_brightness_image(img):
    img = copy.deepcopy(img)
    img = gaussian_filter(img)
    height, width, _ = img.shape#获取shape的数值，height和width、通道    
    img = img.astype(np.float32)
    (b, g, r) = cv2.split(img)
    
    # 将图像均值变换为median
    median_b = 128
    median_g = 128
    median_r = 128
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_shift = b_mean - median_b
    g_shift = g_mean - median_g
    r_shift = r_mean - median_r
    b -= b_shift
    g -= g_shift
    r -= r_shift
    
    # 增强对比度 
    contrast_b = 2
    contrast_g = 2
    contrast_r = 2  
    b = b * contrast_b
    g = g * contrast_g
    r = r * contrast_r
    
    result = cv2.merge((b, g, r))
    
    # 截断越界元素，取整，转换类型
    result[result > 255] = 255
    result[result < 0] = 0
    result = np.round(result)
    result = result.astype(np.uint8)
    return result


# %%
# import cv2
# new_im_paths = glob.glob(os.path.join(image_path,'*.jpg'))

# # 新建文件夹用于保存增强对比度之后的图片
# new_image_path = os.path.join(res_path,'JPEGImages_Contrast')
# if not os.path.exists(new_image_path):
#     os.mkdir(new_image_path)

# for im_path in new_im_paths:
#     print('enhancing %s ....'%im_path, end='\r')
#     img = cv2.imread(im_path)
#     im_Contrast = contrast_brightness_image(img)     
#     cv2.imwrite(im_path.replace('JPEGImages','JPEGImages_Contrast'),im_Contrast)
 

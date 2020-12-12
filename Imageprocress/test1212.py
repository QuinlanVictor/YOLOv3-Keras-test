'''
进行肺实质分割程序的测试

vesion：20200904
author：Quinlan

读取图片信息进行肺实质分割，对于没办法显示的连通区域，可以考虑调整选择区域的标准

1211 在main0904的基础上测试新整理的png数据集
1212 调整得到的阈值分割图

筛选隔板  B[2] < col_size / 5 * 4
'''


import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans



import cv2


# Standardize the pixel values
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=1).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image 满足输出1，不满足输出0

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([5, 5]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    good_area = []
    numsR = len(regions)
    print('regions数量为：', numsR)
    for prop in regions:
        B = prop.bbox
        A = prop.area
        print(A)
        print(B)
        #if A > 3000 and A < 150000:
        if A > 1000 and A < 50000:
            if B[2] - B[0] < row_size / 10 * 10 and B[3] - B[1] < col_size / 10 * 10 and B[0] > row_size / 12  and B[
                2] < col_size  :

        #if A == 19749 or A==18319:
                good_labels.append(prop.label)


    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    # print(good_labels)
    # print(row_size)
    # print(col_size / 5 *4)
    # print(col_size / 10 * 9)
    # print(col_size / 12 )

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask


def raw2mask():
    #int_path = r"E:\Data\LIDC\pngdata1211"
    int_path = r'E:\Files\Repositories\0830\png'
    i = 0
    for root, dirs, files in os.walk(int_path):
        for filename in files:  # 遍历所有文件
            i += 1
            print(filename)
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #mask = make_lungmask(gray, display=True)
            mask = make_lungmask(gray, display=False)
            #Img = np.hstack((gray, mask * gray))
            Img=(mask*gray)
            #cv2.imwrite("E:\Data\LIDC\Segmentationjpg\png1212/" + filename, Img)
            cv2.imwrite("E:\Files\Repositories/0830/result/resultpng/" + filename, Img)
            #
            # if i == 5:
            #     break


if __name__ == '__main__':
    raw2mask()

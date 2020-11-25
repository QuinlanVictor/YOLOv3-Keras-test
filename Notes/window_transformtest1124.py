"""
20201124

对于ct图像进行重采样等预处理的测试代码


"""
import os
import SimpleITK as sitk



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


'''进行重采样的测试'''
import scipy.ndimage as ndimg


'''读取图像数据'''
ct_path = 'E:\Files\Repositories\kerasYolov4/test\dicom/1.dcm'
#ct = sitk.ReadImage(os.path.join(ct_path,'/1.dcm'))
ct = sitk.ReadImage(ct_path)
origin = ct.GetOrigin()
direction = ct.GetDirection()
xyz_thickness = ct.GetSpacing()
ct_array = sitk.GetArrayFromImage(ct)
saved_path = 'E:\Files\Repositories\kerasYolov4/test\dicom'
saved_name = os.path.join(saved_path,'tran1.dcm')

tran = window_transform(ct_array, 1500, -400, normal=False)

saved_preprocessed(tran, origin, direction, xyz_thickness, saved_name)


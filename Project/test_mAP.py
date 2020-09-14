import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

## Import Library
from yolo_my import YOLO, detect_video
from PIL import Image

from PIL import Image
import numpy as np

            
def draw_gt_box(box, image_src):
    image = image_src.copy()
    left, top, right, bottom, class_gt = box 
    label_class = class_id[class_gt]
    label = '{}'.format(label_class)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    c = class_gt
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=(255,255,255))
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=(255,255,255))
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    
    return image

## 2. Get testset
import pandas as pd
annotation_path = 'test.txt'
with open(annotation_path) as f:
    annotation_lines = f.readlines()
#df_testset.columns = ['filename', 'x_start', 'y_start', 'x_end', 'y_end', 'label']

## 3. Get labels
class_label_path = 'model_data/coco_classes.txt'
with open(class_label_path) as f:
    class_id = f.readlines()
class_id = [ele.strip() for ele in class_id]

## 4. Get front
import colorsys
import cv2
class_names = class_id
# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
np.random.seed(255)  # Fixed seed for consistent colors across runs.
np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.

## 5. Args
import argparse
# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--model', type=str,
    help='path to model weight file, default ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--anchors', type=str,
    help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
)

parser.add_argument(
    '--classes', type=str,
    help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
)

parser.add_argument(
    '--gpu_num', type=int,
    help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
)

parser.add_argument(
    '--image', default=False, action="store_true",
    help='Image detection mode, will ignore all positional arguments'
)
'''
Command line positional arguments -- for video detection mode
'''
parser.add_argument(
    "--input", nargs='?', type=str,required=False,default='./path2your_video',
    help = "Video input path"
)

parser.add_argument(
    "--output", nargs='?', type=str, default="",
    help = "[Optional] Video output path"
)

FLAGS = parser.parse_args([])
print(FLAGS)

yolo_detect = YOLO(**vars(FLAGS))  #vars()返回对象object的属性和属性值的字典对象

line = annotation_lines[0].split()
file_path = '/'.join(line[0].split('/')[-4:])
print("Processing: ", file_path)
print()

image = Image.open(file_path)
    
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness = (image.size[0] + image.size[1]) // 300

## 6. Make mAP folder
mAP_folder = 'input'
os.makedirs(mAP_folder, exist_ok = True)

ground_truth = os.path.join(mAP_folder, 'ground-truth')
os.makedirs(ground_truth, exist_ok = True)

detection_results = os.path.join(mAP_folder, 'detection-results')
os.makedirs(detection_results, exist_ok = True)

## 7. Write ground-truth files
def write_ground_truth_files(line, save_folder = 'input/ground-truth'):
    file_path = '/'.join(line[0].split('/')[-4:])
    print(file_path)
    image_file = file_path.split('/')[-1].split('.')[0]

    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    # print(box)

    save_file = os.path.join(save_folder, image_file + '.txt')
    ground_truth_file = open(save_file, 'w')
    for bbox in box:

        left, top, right, bottom, class_gt = bbox 
        class_name = class_id[class_gt]


        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        ground_truth_file.write('%s %d %d %d %d\n'%(class_name, left, top, right, bottom))
    ground_truth_file.close() 

for annotation_line in annotation_lines:
    write_ground_truth_files(annotation_line.split())

## 8. Write detection results
def write_detecion_result_files(line, results, save_folder = 'input/detection-results'):
    file_path = '/'.join(line[0].split('/')[-4:])
    print(file_path)
    image_file = file_path.split('/')[-1].split('.')[0] 

    save_file = os.path.join(save_folder, image_file + '.txt')
    detecion_result_file = open(save_file, 'w')
    for bbox in results:

        class_name_pre, confidence, left, top, right, bottom = bbox 
        


        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        detecion_result_file.write('%s %f %d %d %d %d\n'%(class_name_pre, confidence, left, top, right, bottom))
    detecion_result_file.close() 

for annotation_line in annotation_lines:
    #annotation_line = annotation_lines
    line = annotation_line.split()
    file_path = '/'.join(line[0].split('/')[-4:])
    print("Processing: ", file_path, end='\r')
 
    
    image = Image.open(file_path)
   
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])  #[1:]去掉列表中第一个元素进行操作

    r_image, yolo_detection_res = yolo_detect.detect_image(image)
    write_detecion_result_files(line, yolo_detection_res)
 
 ## 9. Show images
for annotation_line in annotation_lines:
    #annotation_line = annotation_lines
    line = annotation_line.split()
    file_path = '/'.join(line[0].split('/')[-4:])
    print("Processing: ", file_path, end='\r')


    image = Image.open(file_path)

    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    r_image, yolo_detection_res = yolo_detect.detect_image(image)
        
    for i in range(len(box)):         
        r_image = draw_gt_box(box[i], r_image)

    image_res = np.array(r_image)[:,:,(2,1,0)]
    cv2.imshow('res', image_res)
    key = cv2.waitKey(100)   #一个给定时间内等待用户按键触发，必须在有窗口的情况下才行，cv2.waitKey(0)就是一直等着按键
    if key ==27:
        break

cv2.destroyAllWindows()


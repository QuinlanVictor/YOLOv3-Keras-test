import os


import numpy as np
import codecs
import pandas as pd
import json
from glob import glob
from sklearn.model_selection import train_test_split
import argparse


'''
usage:
python voc_train_test_split.py -d VOC2007-imgaug
'''


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-d', '--dataset-path', type=str, default = "VOC2007", help="Folder of dataset")
args = parser.parse_args()
print(args)


saved_path = args.dataset_path + '/'
#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob(saved_path+"Annotations/*.xml")
total_files = [i.replace(os.sep, '/').split("/")[-1].split(".xml")[0] for i in total_files]

print('Spliting %d files'%(len(total_files)))
train_val_files,test_files = train_test_split(total_files,test_size=0.15,random_state=42)
train_files,val_files = train_test_split(train_val_files,test_size=0.15,random_state=42)

#trainval
for file in train_val_files:
    ftrainval.write(file + "\n")

#test
for file in test_files:
    ftest.write(file + "\n")

#train
for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
import time
import os
import shutil
import string
import json

###############################################
'''
生成只有图片路径的txt文件用的脚本
存到f.txt这个文件里
输入是annotation里的json文件
'''
###############################################


json_file='/home/lichengkun/dataset/trunk_highway_coco/annotations/val.json' 
f = open('/home/yuqiushuang/dataset/detection/new_5382.txt','w')

data=json.load(open(json_file,'r'))
abs_path = '/home/lichengkun/dataset/trunk_highway_coco/val/'
for img in data['images']:
    filename = img["file_name"]
    file_info = "%s%s"%(abs_path,filename)
    f.write(file_info)
    f.write('\n')
f.close()


    
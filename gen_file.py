# coding=utf-8
from __future__ import print_function
import  json
import shutil
import os
import sys, zipfile
#####################################################
'''
此脚本可以直接生成 txt文件，每一个图片就是一个文件，内容包含 
<bbox><category>...
'''
#####################################################
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
 
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
 
ori_json_file='/home/yuqiushuang/dataset/detection/TRUNK_highway_20210511_side_coco/annotations/train.json' # # Object Instance 类型的标注

data=json.load(open(ori_json_file,'r'))
label_tuple = ('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')

ana_txt_save_path = "/home/yuqiushuang/dataset/detection/old_train_txt/" #这里存的是数据集里的json文件对应的每一张图片里的bbox
if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)



for img in data['images']:
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    #每一张图片就是一个txt文件
    ans_name = filename.split('jpg')[0] +'txt'
    f_txt = open(os.path.join(ana_txt_save_path, ans_name), 'w')#存放图片路径和类别信息和bbox
    for ann in data['annotations']:
        if ann['image_id']==img_id:
            #annotation.append(ann)
            #print(ann["category_id"], ann["bbox"])
            box = ann["bbox"]
            cat = ann["category_id"]
            x_min = int(box[0])
            y_min =int(box[1])
            x_max = int(box[0]+box[2])
            y_max =  int(box[1]+box[3])
            category = int(cat)
            confidence = 1
            box_info = " %d %d %d %d %d %d %s" % (
                                        x_min, y_min, x_max, y_max, category,confidence,label_tuple[category-1])#左上和右下的坐标
            f_txt.write(box_info)     
            f_txt.write('\n')  

f_txt.close()
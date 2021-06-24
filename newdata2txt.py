from __future__ import print_function
import os

'''
此脚本用于将新数据集中的yolo可读格式转化为预测用的txt文件每一个txt就是一张图片，内容包括：
归一化后的bbox(左上和右下坐标)，categrory_id, confidence , label
'''

def change_bbox(norm_bbox,width,height):
    x_min =( norm_bbox[0] - norm_bbox[2]/2)*width
    x_min = round(x_min)
    x_max =(norm_bbox[0] +norm_bbox[2]/2)*width
    x_max = round(x_max)
    y_min = (norm_bbox[1] - norm_bbox[3]/2)*height
    y_min = round(y_min)
    y_max = (norm_bbox[1] + norm_bbox[3]/2)*height
    y_max = round(y_max)
    b_box = [x_min, y_min, x_max, y_max]
    return b_box

txt_foler = '/home/yuqiushuang/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/labels/'
result_folder ='/home/yuqiushuang/dataset/detection/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/new_00_txt/'
gt_img = os.listdir(txt_foler)
label_tuple = ('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')
img_width= 1280
img_height =720

for gt in gt_img:
    with open(txt_foler+gt,'r') as fp:
        result_txt  = open(os.path.join(result_folder, gt), 'w')
        for line in fp.readlines():
            norm_bbox = line.split()
            center_x = float(norm_bbox[1])
            center_y = float(norm_bbox[2])
            width = float(norm_bbox[3])
            height = float(norm_bbox[4])
            bbox = [center_x,center_y,width,height]
            bbox = change_bbox(bbox,img_width,img_height)
            label = label_tuple[int(norm_bbox[0])]
            category = int(norm_bbox[0])+1
            box_info = " %d %d %d %d %d %d %s" % (
                                       bbox[0],bbox[1],bbox[2],bbox[3],category,1,label)#左上和右下的坐标
            result_txt.write(box_info)     
            result_txt.write('\n')  

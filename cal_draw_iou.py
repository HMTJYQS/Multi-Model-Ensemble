import json
import os
import cv2
from numpy.lib import shape_base

'''
此脚本可以统计labelme格式的模型预测结果文件与gt之间iou的分布情况
'''
def compute_iou(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou


gt_bbox = '/home/yuqiushuang/dataset/detection/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/new_00_txt/'
gt_img_file = '/home/yuqiushuang/dataset/detection/new_00_draw_gt/'
pre_bbox = '/home/yuqiushuang/dataset/detection/badcase/labelme/new_00_wbf_0.95/'
result_file = '/home/yuqiushuang/dataset/detection/badcase/labelme/new_00_wbf0.95_iou0.9/'

gt_img = os.listdir(gt_bbox)
pre_img = os.listdir(pre_bbox)
iou_dict = {}#每一个预测框与gt之间的iou,统计数量
label_iou = {}#每一类在特定iou下的各类框的数量
iou_threshold = 0.9

for pre in pre_img:
    pre_name = pre.split('.')[0]
    img_path = gt_img_file+ pre_name+'.jpg'
    img = cv2.imread(img_path)
    pre_file = pre_bbox+pre
    data=json.load(open(pre_file,'r'))
    for info in data['shapes']:
        label = info['label']
        label = label.split()[0]
        #从图片中取出一个框
        if 'points'in info.keys():
            xtl =info['points'][0][0]
            ytl =  info['points'][0][1]
            xbr = info['points'][1][0]
            ybr = info['points'][1][1]
            bbox_pre = [xtl,ytl,xbr,ybr]
            iou_max = 0
            for gt in gt_img:
                gt_name = gt.split('.')[0]
                #和GT图片挨个比较
                if pre_name == gt_name:                   
                    gt_data = open(gt_bbox+gt,'r')
                    for line in gt_data.readlines():
                        lines = line.split()
                        gt_xtl = int(lines[0])
                        gt_ytl = int(lines[1])
                        gt_xbr = int(lines[2])
                        gt_ybr = int(lines[3])
                        bbox_gt = [gt_xtl,gt_ytl,gt_xbr,gt_ybr]
                        iou = compute_iou(bbox_gt,bbox_pre)
                        if iou >0 :
                            if iou > iou_max:
                                iou_max = iou 
                            else:
                                continue
                    break           
            # if iou_max > iou_threshold:#把所有大于iou阈值的都画出来
            #     cv2.rectangle(img,(int(xtl),int(ytl)),(int(xbr),int(ybr)),(0,255,0),1)
            #     #cv2.putText(img,label,(int(xtl),int(ytl)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)
            #     iou_max = 0
            #     path_name = result_file + pre_name + '.jpg'
            #     cv2.imwrite(path_name,img)   
            # else:
            #     continue
            #在confidence很高的情况下如果出现了与GT之间的iou小于0.6，可以去除这个地方的注释把图片名字打印出来，然后去对应的文件夹
            # 看看GT和prediction的结果，分析一下造成iou这么小的原因是什么 
            # if iou_max <0.6:
            #     print(iou_max, ':', gt_name)   
            if iou_max >iou_threshold:
                #每读完一张gt图上所有的框，找到pre图与之iou最大的框，这就是pre对应的原图上的框，然后统计这个iou是多大
                iou_value = int(iou_max *10)/10
                if str(iou_value) in iou_dict.keys():
                    iou_dict[str(iou_value)] +=1        
                else:
                    iou_dict[str(iou_value)] =1
                if label in label_iou.keys():
                    label_iou[label] += 1
                else:
                    label_iou[label] = 1     
        else:
            continue 
   
print(iou_dict)    
print(label_iou)
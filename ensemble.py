from __future__ import print_function
from ensemble_wbf import weighted_boxes_fusion
import numpy as np
#from ensemble_boxes import *
import itertools
import  json
import os
import sys, zipfile
import cv2
from PIL import Image
import base64
from collections import OrderedDict 
'''
输入数据的格式：
boxes_list = [[
    [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
]]
scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]#各个预测框的置信度
labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
weights = [2, 1]#不同模型的权重

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)

boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
'''


###########################################
'''
此脚本可以将不同网络预测出的图片结果依据bbox,conf进行iou比较，进行数据的预标注
'''
###########################################

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

def change_bbox(norm_bbox):
    x_min =( norm_bbox['center_x'] - norm_bbox['width']/2)*1280
    x_min = round(x_min)
    x_max =(norm_bbox['center_x'] +norm_bbox['width']/2)*1280
    x_max = round(x_max)
    y_min = (norm_bbox['center_y'] - norm_bbox['height']/2)*1080
    y_min = round(y_min)
    y_max = (norm_bbox['center_y'] + norm_bbox['height']/2)*1080
    y_max = round(y_max)
    b_box = [x_min, y_min, x_max, y_max]
    return b_box

def norm_bbox(bbox):
    '''
    input：yolov4预测结果里的中心坐标加宽高，classes['relative_coordinates']的内容
    return：归一化的左上和右下的坐标
    '''
    x_min =( bbox['center_x'] - bbox['width']/2)
    x_max =(bbox['center_x'] +bbox['width']/2)
    y_min = (bbox['center_y'] - bbox['height']/2)
    y_max = (bbox['center_y'] + bbox['height']/2)
    b_box = [x_min, y_min, x_max, y_max]
    return b_box 

def txt2list(txt_file_path):
    '''
    input: txt_file_path图片的txt文件路径,文件内容是没有被归一化的bbox坐标，id,conf 和name
    return: content_list 一个包含归一化的bbox坐标，confidence信息的list
    '''
    content_list = []
    content = open(txt_file_path,'r')#打开图片
    for line in content.readlines():
        line_list = line.split()
        id = line_list[-3]
        id = int(id)
        confidence = float(line_list[-2])#倒数第二位是confidence
        gt_x_min = int(line_list[0])/1280
        gt_y_min = int(line_list[1])/1080
        gt_x_max = int(line_list[2])/1280
        gt_y_max = int(line_list[3])/1080
        name = line_list[-1]
        line_tuple = [gt_x_min,gt_y_min,gt_x_max,gt_y_max,id, confidence,name]
        content_list.append(line_tuple)
    return content_list

def det2json( image_path, object_dict, savepath):
    """
    image_path：图像路径
    object_dict：检测的结果，是一个里面是字典的列表
    savepath:保存的base路径
    """
    image_name = image_path
    json_name = image_path[:-4] + '.json'  # 43_788_533.json
    f = open(image_path, 'rb')
    img = cv2.imread(image_path)
    imageHeight, imageWidth, channel = img.shape
    # 参数image：图像base64编码
    imageData = base64.b64encode(f.read()).decode('utf-8')  # .decode('utf-8')是为了去除字符串前面的r

    # json的前面部分
    data_prex = OrderedDict([("version", "4.2.9"), ("flags", {}), ("shapes", [])])
    json_final = json.dumps(data_prex, skipkeys=False, ensure_ascii=False, sort_keys=False, indent=2)

    # json_final表示最终的json
    json_final = json_final[:-3]
    # print(json_final)
    if len(object_dict) == 0:
        print("no label in this {}, skipped~".format(image_name))
        return
    for one_object in object_dict:
        label = one_object['name']
        label = label +' '+'confidence:'+ str(one_object['confidence'])
        box_tmp = one_object['relative_coordinates']
        box = [
            # float(box_tmp["center_x"]) - float(box_tmp["width"]) / 2.),
            box_tmp["center_x"] - box_tmp["width"] / 2.,
            box_tmp["center_y"] - box_tmp["height"] / 2.,
            box_tmp["center_x"] + box_tmp["width"] / 2.,
            box_tmp["center_y"] + box_tmp["height"] / 2.
        ]
        # OrderedDict不然保存的顺序会乱
        # @###注意！！因为"line_color", null后面的引号，我去不掉，就改了labelme的源码
        # 注释掉了app.py中的   #if line_color:#if fill_color:
        #想根据置信度的大小画不同颜色的框，所以加了这个if-else
        if one_object['confidence'] >0.5 :
            data_cen = OrderedDict([("label", label), ("line_color", [0,0,255]), ("fill_color", 'null'),
                                    ("points", [[box[0] * imageWidth, box[1] * imageHeight],
                                                [box[2] * imageWidth, box[3] * imageHeight]]),
                                    ("shape_type", "rectangle"),
                                    ("flags", {})
                                    ])
        else:
            data_cen = OrderedDict([("label", label), ("line_color",'null'), ("fill_color", 'null'),
                                    ("points", [[box[0] * imageWidth, box[1] * imageHeight],
                                                [box[2] * imageWidth, box[3] * imageHeight]]),
                                    ("shape_type", "rectangle"),
                                    ("flags", {}),("confidence",one_object['confidence'])
                                    ])
        json_final += json.dumps(data_cen, skipkeys=False, ensure_ascii=False, sort_keys=False, indent=4)
        json_final += ','

    # print(json_final)
    json_final = json_final[:-1] + ']' + ','
    data_final = OrderedDict([("imagePath", os.path.basename(image_path)),
                                ("imageData", str(imageData)), ("imageHeight", imageHeight),
                                ("imageWidth", imageWidth)])
    json_final += json.dumps(data_final, skipkeys=False, ensure_ascii=False, sort_keys=False, indent=2)[1:]
    # print(json_final)
    with open(os.path.join(savepath, os.path.basename(json_name)), 'w', encoding='gbk') as f:  # wondow要用要用gbk编码
        f.write(json_final)

#计算相关的评价指标
def calc_detection_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
 
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)
 
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
 
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):
        """先对每个图片进行循环"""
 
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
 
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            """循环每个类别"""
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
 
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]
 
            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)
 
            if len(pred_bbox_l) == 0:#如果没有预测这个类别的 直接循环下一个类别了
                continue
            if len(gt_bbox_l) == 0:#如果gt没有这个类别，还预测了这个类别，match填入0，没匹配上
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
 
            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
 
            iou = bbox_iou(pred_bbox_l, gt_bbox_l)#对于每个图片，类别正确 才开始计算iou，iou>阈值 才说明正确
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou
 
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index: #这里是每个gt只匹配一次，下面图片说明
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
 
    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')
 
    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
 
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
 
        order = score_l.argsort()[::-1]
        match_l = match_l[order]
 
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
 
        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
 
    return prec, rec


ori_json_file='/home/yuqiushuang/dataset/detection/badcase/old_side_train_result.json' #model1的预测结果
ori2_json_file = '/home/lichengkun/dataset/TRUNK_highway_20210511_side_coco/infer/train/'#model2的预测结果,每一张图片一个txt文件
final_result  = '/home/yuqiushuang/dataset/detection/badcase/badcase_labelme_nms/'#错误标注图片存放的路径

data=json.load(open(ori_json_file,'r'))
prediction2  = os.listdir(ori2_json_file)
total_files = len(data)
label_tuple = ('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')

# pred_bboxes,pred_labels,pred_scores = list(),list(),list()
# gt_bboxes,gt_labels = list(),list()
# gt_difficults = None

#data是预测图片结果的json文件
for img in data:
    frame_id = img['frame_id']
    filename = img['filename']
    img_name = filename.split('/')[-1]

    object = img['objects']

    #model1预测出的bbox信息
    b_box_list1 = []
    label1 = []
    conf1 = []

    #将object里的一个class_id拿出来
    for classes in object:
        #只有confidence大于0.5的才能有gt存在，其余都是存疑的
        class_id = classes['class_id'] + 1###检测出的bbox的ID 
        label1.append(class_id)
        bbox = classes['relative_coordinates']####这里是归一化之后的坐标
        b_box = norm_bbox(bbox)#变成了归一化的左上和右下的坐标
        b_box_list1.append(b_box)
        conf1.append(classes['confidence'])

    #model2的预测结果，从里面拿出来图片的txt文件
    for file in prediction2:
        file_name = file.split('.')[0] + '.jpg'
        if img_name == file_name:
            txt_file_path = ori2_json_file+file#图片txt的绝对路径
            #把图片txt变成一个list
            content_list = txt2list(txt_file_path)#
            b_box_list2 = []#model2预测出的bbox信息
            label2 = []
            conf2 = []
            for i in content_list:
                b_box_list2.append(i[:4])
                label2.append(i[4])
                conf2.append(1)

            # 进行多模型融合ensemble 
            boxes_list = [b_box_list1,b_box_list2]
            scores_list = [conf1,conf2]
            labels_list = [label1,label2]
            weights = [1,2]

            iou_thr = 0.6
            skip_box_thr = 0.0001
            sigma = 0.1
            
            #boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

            # #将预测的目标框，类别，分数存入list
            # bounding1 = np.expand_dims(boxes, axis=0)
            # confidence1 = np.expand_dims(scores,axis=0)
            # labels1 = np.expand_dims(labels,axis=0)
            # pred_bboxes += list(bounding1)
            # pred_labels += list(labels1)
            # pred_scores += list(confidence1)

            # #将真实的目标框、类别、difficults存入list
            # bbox1 = np.expand_dims(b_box_list2,axis=0)
            # label_true = np.expand_dims(label2,axis=0)
            # gt_bboxes += list(bbox1)
            # gt_labels += list(label_true)

            iou_dict = []
            result_box = []
            
            #把预测框转化为labelme需要的[center_x,center_y,width, height]的格式
            for i in boxes:
                center_x = (i[0]+i[2])/2
                center_y =  (i[1]+i[3])/2
                width = i[2]-i[0]
                height = i[3]-i[1]
                result_bbox = [center_x,center_y,width,height]
                result_box.append(result_bbox)
            for result in zip(labels,result_box,scores):
                center_x = result[1][0]
                center_y = result[1][1]
                width = result[1][2]
                width =width.astype(np.float64)
                height = result[1][3]
                height = height.astype(np.float64)
                class_id = int(result[0])
                name = label_tuple[class_id - 1] 
                confidence = result[2]
                confidence = confidence.astype(np.float64)
                rest_dict = {'class_id':class_id,'name':name,'relative_coordinates':{'center_x':center_x,'center_y':center_y,'width':width,'height':height},'confidence':confidence}
                iou_dict.append(rest_dict)

            #拿到一张图的就把一张图转化为labelme的可读格式
            det2json(filename, iou_dict, final_result) # fixme change the result and savpath params
    print("processing progress is {} %".format(frame_id* 1.0 / total_files * 100))

pre, rec = calc_detection_prec_rec(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults=None,iou_thresh=0.6)

print('pre :',pre)
print('rec :', rec)



                            


                        
        


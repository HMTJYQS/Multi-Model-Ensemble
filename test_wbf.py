from __future__ import print_function
from ensemble_wbf import weighted_boxes_fusion
import numpy as np
#from ensemble_boxes import *
import itertools
import  json
import os
import cv2
from PIL import Image
import base64
from collections import OrderedDict,defaultdict
import six.moves
from cal_eval_index import txt_convert
from numpy.lib.utils import info
from cal_eval_index import json_convert
from xml.etree import ElementTree      # 导入ElementTree模块
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import glob
from xml.dom.minidom import Document

'''
boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
'''


###########################################
'''
此脚本可以利用WBF进行多模型融合，计算融合结果与GT之间的pre和rec,
可输出的labelme格式文件，输出的是设置了prestrain_confidence,
不能输出某一confidence下不同iou的框各有多少个，若想输出运行cal_draw_iou.py
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

def txt2list(txt_file_path,width,height):
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
        gt_x_min = int(line_list[0])/width
        gt_y_min = int(line_list[1])/height
        gt_x_max = int(line_list[2])/width
        gt_y_max = int(line_list[3])/height
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

def bbox_iou(bbox_a, bbox_b):
    '''
    #传入的是真值标签和预测标签  二维数组
    '''
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
 
    # top left  这边是计算了如图上第一幅的重叠左下角坐标值（x，y）
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right  这边是计算了如图上第一幅的重叠左上角坐标值ymax和右下角坐标值xmax
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
 
    #np.prod 给定轴数值的乘积   相减就得到高和宽 然后相乘
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)#重叠部分面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

#计算相关的评价指标pre和rec
def calc_detection_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,iou_thresh,
        gt_difficults=None):
 
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)
 
    n_pos = defaultdict(int)#是个字典，key是label，value是该类的gt有多少个
    score = defaultdict(list)#是个字典，key是label，value是预测框的confidence
    match = defaultdict(list)#是个字典，key是label，value是预测框是否匹配上，匹配上该框就是TP，值为1，否则为0
 
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
            order = pred_score_l.argsort()[::-1]#从大到小给数组排序，返回的是数组中数字的索引，这里conf的第一个值大于第二个值，所以返回的索引就是0，1
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
 
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]
 
            n_pos[l] += np.logical_not(gt_difficult_l).sum()#表示该类的gt一共有多少个
            score[l].extend(pred_score_l)
 
            if len(pred_bbox_l) == 0:#如果没有预测这个类别的 直接循环下一个类别了
                continue
            if len(gt_bbox_l) == 0:#如果gt没有这个类别，还预测了这个类别，match填入0，没匹配上
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
 
            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            #pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
           # gt_bbox_l[:, 2:] += 1
 
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
                        if not selec[gt_idx]:#用逻辑运算在说明预测框是否和gt匹配上
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
    prec = [None] * n_fg_class#表示每个类的precision，其中每个类的precision计算是累加的要注意，为了label和下标的统一，所以下标为0的就是None
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
        prec[l] = tp / (fp + tp)#prec[l]表示的是每增加一个预测框，就会重新计算一个precision，所以precision是累加的
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            
    pre_category = []
    for  line in prec:
        if type(line) == type(None) or len(line) == 0 :
            pre_category.append(0)
            continue
        else:
            precision = line[-1]
            pre_category.append(precision)
        
    rec_category = []
    for  line in rec:
        if type(line) == type(None) or len(line) == 0 :
            rec_category.append(0)
            continue
        else:
            recall = line[-1]
            rec_category.append(recall)
        
    return prec,rec, pre_category[1:], rec_category[1:]

def cal_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    #print(mpre)
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
 
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #print(mpre)
  return ap

def add_txt_model(ori_txt_file, img_name,img_width, img_height, classes= ('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')):
    '''
    input: 模型预测生成的图片txt所在的文件夹路径，待读取的图片名，classes预测的数据集一共有多少类
    output：WBF所需要的数据。是归一化之后的坐标等信息

    '''
    prediction = os.listdir(ori_txt_file)

    #model的预测结果，从里面拿出来图片的txt文件
    for file in prediction:
        file_name = file.split('.')[0] + '.jpg'
        #新的数据集需要转的时候要把这个注释去掉
        #file_name = 'images_'+file_name
        if img_name == file_name:
            txt_file_path = ori_txt_file+file#图片txt的绝对路径
            #把图片txt变成一个list
            content_list = txt2list(txt_file_path,img_width,img_height)
            
            b_box_list = []#model2预测出的bbox信息
            label = []
            conf = []
            for i in content_list:
                b_box_list.append(i[:4])
                label.append(i[4])
                conf.append(i[-2])
            break

    return b_box_list, label, conf

#data是预测图片结果的json文件
def add_json_model(ori_json_file,img_name):
    '''
   input: yolov4预测生成的json文件，yolov4本身生成的就是归一化之后的坐标，所以这里不需要输入数据集里图像的宽高
            
   output: WBF所需要的数据。是归一化之后的坐标等信息
    '''
    data=json.load(open(ori_json_file,'r'))
    for img in data:
        filename = img['filename']
        file_name = filename.split('/')[-1]
       
        if img_name == file_name:
            object = img['objects']
            #model1预测出的bbox信息
            b_box_list1 = []
            label1 = []
            conf1 = []
            #将object里的一个class_id拿出来
            for classes in object:
                class_id = classes['class_id'] + 1###检测出的bbox的ID 

                #为了只提取yolov4检测结果中的前三类添加下面的if语句，（这里根据情况可以自行修改）
                #if class_id < 4:
                bbox = classes['relative_coordinates']##这里是归一化之后的坐标
                if bbox['width'] > 1 or bbox['height']>1:
                    continue
                else:
                    label1.append(class_id)
                    b_box = norm_bbox(bbox)#变成了归一化的左上和右下的坐标
                    b_box_list1.append(b_box)
                    conf1.append(classes['confidence'])
    return b_box_list1,label1,conf1

#生成不同限制条件下的wbf预测框，用于后续计算precision
def constrain_confidence(wbf_boxes,wbf_scores,wbf_labels,prestrain_conf = None): 
    '''
    input:输入的是wbf的输出的所有结果，以及conf的条件
    output: 用于计算precison 的pred_bboxes,pred_labels,pred_scores
    '''
    pred_bboxes,pred_labels,pred_scores = list(),list(),list()
    boxes_copy = []
    scores_copy = []
    labels_copy = []
    wbf_bboxes_count = 0#wbf直接输出的框的数量
    wbf_conf_count = 0#限制了confidence之后wbf框的数量
    #从所有图片中拿出一张图片的 boxes,scores,labels信息
    for boxes,scores,labels in zip(wbf_boxes,wbf_scores,wbf_labels):
        #不限制confidence，直接输出各类precision和recall
        wbf_bboxes_count += len(boxes)
         #如果要统计wbf的结果 ，需要额外添加参数并把这个注释去掉
        if not  prestrain_conf:
            for box,confidence,label in zip(boxes,scores,labels):
            #统计wbf的预测框中每一类的预测框个数
                if label in wbf_label_count_dict.keys():
                    wbf_label_count_dict[label] += 1
                else:
                    wbf_label_count_dict[label] = 1
            boxes_copy = boxes
            scores_copy = scores
            labels_copy = labels
        else:
            for box,confidence,label in zip(boxes,scores,labels):
                #这里由于confidence的原因，boxes_copy等可能为空，但是gt里每一张图还是会读，gt_bboxes不会为空，所以gt_boxes_count每次算出来都是一样的
                if confidence >prestrain_conf:
                    wbf_conf_count += 1
                    boxes_copy.append(box)
                    scores_copy.append(confidence)
                    labels_copy.append(label)
                    #统计wbf的预测框中符合限制条件的每一类的预测框个数
                    if label in wbf_label_count_dict.keys():
                        wbf_label_count_dict[label] += 1
                    else:
                        wbf_label_count_dict[label] = 1
            #新的boxes_copy,scores_copy,labels_copy生成之后才能添加到pre_bboxes等里面
        bounding1 = np.expand_dims(boxes_copy, axis=0)
        confidence1 = np.expand_dims(scores_copy,axis=0)
        labels1 = np.expand_dims(labels_copy,axis=0)
        pred_bboxes += list(bounding1)
        pred_labels += list(labels1)
        pred_scores += list(confidence1)
        boxes_copy = []
        scores_copy = []
        labels_copy = []
    return pred_bboxes,pred_labels,pred_scores,wbf_bboxes_count,wbf_conf_count,wbf_label_count_dict

def cov2labelme(boxes,scores,labels, filename,labelme_json,prestrain_conf=None):
    '''
    input: 每一张wbf融合完的结果图
    output：已经转成了labelme json格式的文件，存在了相应位置
    '''
    boxes_copy = []
    scores_copy = []
    labels_copy = []
    
    if not  prestrain_conf:
        #如果要统计wbf的结果 ，需要额外添加参数并把这个注释去掉
        #wbf_bboxes_count += len(boxes)
        boxes_copy = boxes
        scores_copy = scores
        labels_copy = labels
    else:
        for box,confidence,label in zip(boxes,scores,labels):
            #这里由于confidence的原因，boxes_copy等可能为空，但是gt里每一张图还是会读，gt_bboxes不会为空，所以gt_boxes_count每次算出来都是一样的
            if confidence >prestrain_conf:
                boxes_copy.append(box)
                scores_copy.append(confidence)
                labels_copy.append(label)    
            else:
                continue

    #把预测框转化为labelme需要的[center_x,center_y,width, height]的格式，和boxes_copy放在同一层循环
    iou_dict = []
    result_box = []
    for i in boxes_copy:
        center_x = (i[0]+i[2])/2
        center_y =  (i[1]+i[3])/2
        width = i[2]-i[0]
        height = i[3]-i[1]
        result_bbox = [center_x,center_y,width,height]
        result_box.append(result_bbox)
    #这层for走完就说明读完了一张图
    for result in zip(labels_copy,result_box,scores_copy):
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
        iou_dict.append(rest_dict)#要被转成labelme格式的预测框

    #拿到一张图的就把一张图转化为labelme的可读格式
    det2json(filename, iou_dict, labelme_json) # fixme change the result and savpath params
    print("processing progress is {} %".format(frame_id* 1.0 / total_files * 100))
  
#生成xml文件
def generate_xml(height,width,threshold,json_file_path,img_path,save_path):
    #在内存中创建一个空的文档
    doc = Document() 

    #创建一个根节点Managers对象
    root = doc.createElement('annotations') 

    #设置根节点的属性
        # root.setAttribute('company', '哈哈科技') 
        # root.setAttribute('address', '科技软件园') 

    #将根节点添加到文档对象中
    doc.appendChild(root)
    #添加<version>1.1</version>
    nodeVersion = doc.createElement('version')
    nodeVersion.appendChild(doc.createTextNode(str(1.1)))
    root.appendChild(nodeVersion)
    #添加meta
    nodemeta = doc.createElement('meta')
    nodeLabels = doc.createElement('labels')
#---------------------------------------------------------------------------------------
    #label的属性设置(如果有新添加的label，加在这个字典里面)
#--------------------------------------------------------------------------------------- 
    labelsList = [{'name' : 'truck',  'color' : '#32b7fa'},
                {'name' : 'car',  'color' : '#ddff33'},
                {'name' : 'traffic_sign',  'color' : '#b83df5'},
                {'name' : 'traffic_cone',  'color' : '#b83df5'},
                {'name' : 'traffic_light',  'color' : '#b83df5'},
                {'name' : 'motorbike',  'color' : '#b83df5'},
                {'name' : 'person',  'color' : '#b83df5'},
                {'name' : 'bicycle',  'color' : '#b83df5'},
                {'name' : 'tricycle',  'color' : '#b83df5'},]

    for i in labelsList :
        nodeLabel = doc.createElement('label')
        for j in i:
            #给叶子节点name设置一个文本节点，用于显示文本内容
            nodeName = doc.createElement(j)
            nodeName.appendChild(doc.createTextNode(str(i[j])))
            nodeLabel.appendChild(nodeName)


        nodeAttributes = doc.createElement('attributes')
        nodeAttribute = doc.createElement('attribute')
        nodeName = doc.createElement('name')
        nodeName.appendChild(doc.createTextNode('semi_label_confidence'))
        #将各叶子节点添加到父节点meta中
        nodeAttribute.appendChild(nodeName)

        nodeMutable = doc.createElement('mutable')
        nodeMutable.appendChild(doc.createTextNode('False'))
        nodeAttribute.appendChild(nodeMutable)

        nodeInput_type = doc.createElement('input_type')
        nodeInput_type.appendChild(doc.createTextNode('Select'))
        nodeAttribute.appendChild(nodeInput_type)

        nodeDefault_value = doc.createElement('default_value')
        nodeDefault_value.appendChild(doc.createTextNode('true'))
        nodeAttribute.appendChild(nodeDefault_value)

        nodeValues = doc.createElement('values')
        nodeValues.appendChild(doc.createTextNode('truefalse'))
        nodeAttribute.appendChild(nodeValues)
        nodeAttributes.appendChild(nodeAttribute)

        nodeLabel.appendChild(nodeAttributes)
        nodeLabels.appendChild(nodeLabel)
    nodemeta.appendChild(nodeLabels)

    #添加segments属性
    nodeSegments = doc.createElement('segments')
    nodeSegment = doc.createElement('segment')

    nodeId = doc.createElement('id')
    nodeId.appendChild(doc.createTextNode('16'))
    nodeSegment.appendChild(nodeId)

    nodeStart = doc.createElement('start')
    nodeStart.appendChild(doc.createTextNode('0'))
    nodeSegment.appendChild(nodeStart)

    nodeStop = doc.createElement('stop')
    nodeStop.appendChild(doc.createTextNode('2'))
    nodeSegment.appendChild(nodeStop)

    nodeUrl = doc.createElement('url')
    nodeUrl.appendChild(doc.createTextNode('http://192.168.3.102:1000/?id=16'))
    nodeSegment.appendChild(nodeUrl)
    nodeSegments.appendChild(nodeSegment)
    nodemeta.appendChild(nodeSegments)

    nodeOwner = doc.createElement('owner')

    nodeUsername = doc.createElement('username')
    nodeUsername.appendChild(doc.createTextNode('django'))
    nodeOwner.appendChild(nodeUsername)

    nodeEmail = doc.createElement('email')
    nodeEmail.appendChild(doc.createTextNode('django@trunk.tech'))
    nodeOwner.appendChild(nodeEmail)
    nodemeta.appendChild(nodeOwner)

    nodeAssignee = doc.createElement('assignee')
    nodeAssignee.appendChild(nodeAssignee)
        
    nodeDumped = doc.createElement('dumped')
    nodeDumped.appendChild(doc.createTextNode('2021-06-17 07:57:55.027126+00:00'))
    nodeDumped.appendChild(nodeDumped)
    #nodemeta.appendChild(nodeDumped)

    #最后将meta添加到根节点annotations中
    root.appendChild(nodemeta)

    #添加图片信息及bbox信息
    #根据待标注的数据集来生成xml,因为在生成json文件的时候，如果没有框，这个图片就会被跳过不会生成json文件
    #所以json文件的数量<=图片的数量，所以以图片的数量为准，遇到没有标注的图片，xml里就不加box信息即可
    total_jpg_files = sorted(glob.glob(img_path+'*.jpg'))
    count = 0
    for img_file in total_jpg_files:
        json_name = img_file.split("/")[-1].split('.')[0]+'.json'
        img_name =  img_file.split("/")[-1]
        json_file = json_file_path+json_name
        
        #image节点增加属性
        nodeImage = doc.createElement('image')
        nodeImage.setAttribute('height',str(height))
        nameT=doc.createTextNode('')
        nodeImage.appendChild(nameT)
        nodeImage.setAttribute('id',str(count))
        count+= 1
        nameT=doc.createTextNode('')
        nodeImage.appendChild(nameT)
        nodeImage.setAttribute('name',img_name)
        nodeImage.appendChild(nameT)
        nodeImage.setAttribute('width',str(width))
        #判断json文件夹里是否有这个图片的json文件
        if os.path.isfile(json_file):
            data=json.load(open(json_file,'r'))
        
            #添加bbox坐标信息
            for shapes in data['shapes']:
            #shapes里有好几个bbox，一一拿出来
                nodeBox = doc.createElement('box')
                label = shapes['label'].split()[0]
                confidence =  shapes['label'].split()[1]
                confidence = float(confidence.split(':')[1])
                nodeBox.setAttribute('label',label)
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('occluded','0')
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('source','manual')
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('xtl',str(shapes['points'][0][0]))
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('ytl',str(shapes['points'][0][1]))
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('xbr',str(shapes['points'][1][0]))
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('ybr',str(shapes['points'][1][1]))
                nodeBox.appendChild(nameT)
                nodeBox.setAttribute('z_order','0')

                nodeAttribute = doc.createElement('attribute')
                nodeAttribute.setAttribute('name', 'semi_label_confidence')
                #这里设置是否预json_file_path标注的条件，目前只是卡WBF输出的confidence，之后如果改了别的条件，可以在这里该if里的内容
                #而且这里如果是放的转成labelme的结果 那已经限定过confidence了，这个threshold实属多此一举
                if confidence >= threshold:
                    nodeAttribute.appendChild(doc.createTextNode('true'))
                else:
                    nodeAttribute.appendChild(doc.createTextNode('false'))
                nodeBox.appendChild(nodeAttribute)
                nodeImage.appendChild(nodeBox)##########这里一会试一下，所有框都写完再加到image节点上，现在是每写一个bbox就加到image节点上
                root.appendChild(nodeImage)
        else:
            root.appendChild(nodeImage)
    #开始写xml文档
    fp = open(save_path, 'w',encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()

#美化xml文件
def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行  
    if element:  # 判断element是否有子元素  
        if element.text == None or element.text.isspace(): # 如果element的text没有内容  
            element.text = newline + indent * (level + 1)    
        else:  
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)  
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行  
    #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level  
    temp = list(element) # 将elemnt转成list  
    for subelement in temp:  
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致  
            subelement.tail = newline + indent * (level + 1)  
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个  
            subelement.tail = newline + indent * level  
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作  
    return element

#new_dataset
ori_json_file='/home/yuqiushuang/dataset/detection/badcase/new_00_result.json' #model1的预测结果，这里model1是yolov4的预测结果是json格式文件
ori2_txt_file = '/home/lichengkun/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/infer/train_cascade/'#model2的预测结果,每一张图片一个txt文件
ori3_txt_file = '/home/lichengkun/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/infer/train_gfl3/' #model3的预测结果,每一张图片一个txt文件
gt_txt_file = '/home/yuqiushuang/dataset/detection/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/new_00_txt/'
img_file = '/home/yuqiushuang/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/JPEGImages/'
xml_save_path = '/home/yuqiushuang/dataset/detection/badcase/new_1338.xml'

gt_txt = os.listdir(gt_txt_file)
total_files = len(gt_txt)
label_tuple = ('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')

#数据集参数设置
img_width = 1280
img_height = 720
prestrain_conf = 0.5
xml_conf = 0.98
iou_thresh = 0.9#有GT的时候计算precision需要输入的参数
weights = [1,2,2]
labelme_json  = '/home/yuqiushuang/dataset/detection/badcase/labelme/new_00_wbf_cascade_0.5/'#labelme标注图片存放的路径

#WBF参数设置
iou_thr = 0.6
skip_box_thr = 0.0001
sigma = 0.1

#待输出的各类指标
gt_bboxes,gt_labels = list(),list()
##因为这里是图片全部进行完WBF之后再统一转化为labelme格式文件，所以这里设置了以下三个list用于存放所有融合完的图片
wbf_boxes = []
wbf_labels = []
wbf_scores = []
gt_difficults = None# precision需要的参数，占位用
gt_bboxes_count = 0#统计gt一共多少个框
gt_label_count_dict = {}#统计gt中每一类的预测框个数
wbf_label_count_dict = {}#统计wbf的预测框中符合限制条件的每一类的预测框个数
frame_id = 0#计算processing时要用

#-------------------------------------------------------------------------------------------
#数据集里的bbox等信息
#-------------------------------------------------------------------------------------------
for gt in gt_txt:
    frame_id += 1
    img_name = gt.split('.')[0] + '.jpg'
    #如果想要单独计算单张图片，可以在这行注释的下面加上个if语句
    txt_file_path = gt_txt_file+gt#图片txt的绝对路径
    img_file_path = img_file+img_name
    #把图片txt变成一个list
    content_list = txt2list(txt_file_path,img_width,img_height)
    
    b_box_gt = []#gt的bbox信息
    gt_label = []
    gt_conf = []
    for i in content_list:
        b_box_gt.append(i[:4])
        gt_label.append(i[4])
        #计算gt里面各个种类的预测框的数量
        if i[-1] in gt_label_count_dict.keys():
            gt_label_count_dict[i[-1]] += 1
        else:
            gt_label_count_dict[i[-1]] = 1
        gt_conf.append(1)
        gt_bboxes_count += 1

    #将真实的目标框、类别、difficults存入list
    bbox1 = np.expand_dims(b_box_gt,axis=0)
    label_true = np.expand_dims(gt_label,axis=0)
    gt_bboxes += list(bbox1)
    gt_labels += list(label_true)    
#-------------------------------------------------------------------------------------------
#  添加多个模型预测的结果，进行多模型融合
#-------------------------------------------------------------------------------------------
    #model1,2,3预测出的bbox,label,confidence等信息
    b_box_list1,label1,conf1 = add_json_model(ori_json_file,img_name)
    b_box_list2,label2,conf2 = add_txt_model(ori2_txt_file,img_name,img_width,img_height)
    b_box_list3,label3,conf3 = add_txt_model(ori3_txt_file,img_name,img_width,img_height)

    # 进行多模型融合ensemble 
    boxes_list = [b_box_list1,b_box_list2,b_box_list3]
    scores_list = [conf1,conf2,conf3]
    labels_list = [label1,label2,label3]
   

    
    #boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    #boxes是归一化后的左上和右下坐标
    print('img_name:',img_name)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    wbf_boxes.append(boxes)
    wbf_scores.append(scores)
    wbf_labels.append(labels)
#-------------------------------------------------------------------------------------------
#  一次将一张图片转化为labelme格式文件,可以选择输出限制confidence为多少的那些框
#-------------------------------------------------------------------------------------------
    cov2labelme(boxes,scores,labels, img_file_path ,labelme_json,prestrain_conf)
#-------------------------------------------------------------------------------------------
#将数据集全部转化为预标注的xml文件
#-------------------------------------------------------------------------------------------
generate_xml(img_height,img_width,xml_conf,labelme_json,img_file,xml_save_path)
#-------------------------------------------------------------------------------------------
#设置限制precision的条件,计算precision和recall
#-------------------------------------------------------------------------------------------
 #输出precision计算需要的pred_bbox等
# pred_bboxes,pred_labels,pred_scores,wbf_bboxes_count,wbf_conf_count,wbf_label_count_dict = constrain_confidence(wbf_boxes,wbf_scores,wbf_labels,prestrain_conf)
#  #计算pre rec,放在for img in data外面
# prec,rec, pre_category, rec_category= calc_detection_prec_rec(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,iou_thresh, gt_difficults=None)
    
# print('pre :',pre_category)
# print('rec :', rec_category)
# print('wbf一共预测出来多少框:',wbf_bboxes_count)#统计wbf一共预测出多少个框
# print('wbf{}框的数量:'.format(prestrain_conf),wbf_conf_count)
# print('gt总框数',gt_bboxes_count)
# print('gt每一类的总数量:',gt_label_count_dict)
# print('wbf结果conf:{}每一类框的数量:'.format(prestrain_conf),wbf_label_count_dict)



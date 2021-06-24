from __future__ import print_function

from numpy.lib.function_base import append
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

###########################################
'''
此脚本计算融合之前某一模型预测结果与GT之间的pre,rec和ap
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
    '''
    把归一化的坐标转化为正常的坐标
    input:norm_bbox 归一化的坐标
    output: 正常的坐标
    '''
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
        if type(line) == type(None) or len(line) == 0:
            pre_category.append(0)
            continue
        else:
            
            precision = line[-1]
            pre_category.append(precision)
        
    rec_category = []
    for  line in rec:
        if type(line) == type(None) or  len(line) == 0:
            rec_category.append(0)
            continue
        else:
            recall = line[-1]
            rec_category.append(recall)
        
    return prec,rec,pre_category[1:], rec_category[1:]

def json_convert(ori_json_file,gt_file,classes=('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle')):
    '''
    input:yolov4的json文件绝对路径 ，gt_file的相对路径，gt_file是一个有所有图片txt的文件夹，classes是tuple类型的
    output：pred_bbox,pred_labels, pred_confidence,gt_bbox,gt_labels 

    注：因为在计算precision的时候都是和gt进行比较的，所以要同时输出gt的信息
    '''
    data=json.load(open(ori_json_file,'r'))
    total_files = len(data)
    gt_img = os.listdir(gt_file)

    #用于计算precision和recall的参数
    pred_bboxes,pred_labels,pred_scores = list(),list(),list()
    gt_bboxes,gt_labels = list(),list()
    gt_difficults = None

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
            class_id = classes['class_id'] + 1###检测出的bbox的ID 
            bbox = classes['relative_coordinates']####这里是归一化之后的坐标
            if bbox['width'] > 1 or bbox['height']>1:
                continue
            else:
                label1.append(class_id)
                b_box = norm_bbox(bbox)#变成了归一化的左上和右下的坐标
                b_box_list1.append(b_box)
                conf1.append(classes['confidence'])

        #将预测的目标框，类别，分数存入list
        bounding1 = np.expand_dims(b_box_list1, axis=0)
        confidence1 = np.expand_dims(conf1,axis=0)
        labels1 = np.expand_dims(label1,axis=0)
        pred_bboxes += list(bounding1)
        pred_labels += list(labels1)
        pred_scores += list(confidence1)  

        #ground truth的目标框，类别和分数信息
        for gt in gt_img:
            file_name = gt.split('.')[0] + '.jpg'
            
            if img_name == file_name:
                txt_file_path = gt_file+gt#图片txt的绝对路径
                #把图片txt变成一个list
                content_list = txt2list(txt_file_path)
                b_box_gt = []#model2预测出的bbox信息
                gt_label = []
                gt_conf = []
                for i in content_list:
                    b_box_gt.append(i[:4])
                    gt_label.append(i[4])
                    gt_conf.append(1)

            #将真实的目标框、类别、difficults存入list
                bbox1 = np.expand_dims(b_box_gt,axis=0)
                label_true = np.expand_dims(gt_label,axis=0)
                gt_bboxes += list(bbox1)
                gt_labels += list(label_true)
                break
    return pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults

def txt_convert(ori_txt_file,gt_file, classes =('car','truck','bus','traffic_sign','traffic_cone','traffic_light','motorbike','person','bicycle','tricycle') ):
    '''
    input:模型预测的图片txt所在的文件夹路径 ，gt_file的相对路径，gt_file是一个有所有图片txt的文件夹，classes是tuple类型的
    output：pred_bbox,pred_labels, pred_confidence,gt_bbox,gt_labels 
    '''
    prediction  = os.listdir(ori_txt_file)
    gt_img = os.listdir(gt_file)

    #用于计算precision和recall的参数
    pred_bboxes,pred_labels,pred_scores = list(),list(),list()
    gt_bboxes,gt_labels = list(),list()
    gt_difficults = None

    #模型预测结果
    for img in prediction:
        img_name = img.split('.')[0] + '.jpg'
        model_path = ori_txt_file+img#图片txt的绝对路径
        #把图片txt变成一个list
        model_content_list = txt2list(model_path)

        #model1预测出的bbox信息
        b_box_list1 = []
        label1 = []
        conf1 = []
        for i in model_content_list:
            b_box_list1.append(i[:4])
            label1.append(i[4])
            conf1.append(i[-2])

        #将预测的目标框，类别，分数存入list
        bounding1 = np.expand_dims(b_box_list1, axis=0)
        confidence1 = np.expand_dims(conf1,axis=0)
        labels1 = np.expand_dims(label1,axis=0)
        pred_bboxes += list(bounding1)
        pred_labels += list(labels1)
        pred_scores += list(confidence1)      

    #ground truth的目标框，类别和分数信息
        for gt in gt_img:
            file_name = gt.split('.')[0] + '.jpg'
            
            if img_name == file_name:
                txt_file_path = gt_file+gt#图片txt的绝对路径
                #把图片txt变成一个list
                content_list = txt2list(txt_file_path)
                b_box_gt = []#model2预测出的bbox信息
                gt_label = []
                gt_conf = []
                for i in content_list:
                    b_box_gt.append(i[:4])
                    gt_label.append(i[4])
                    gt_conf.append(1)

                #将真实的目标框、类别、difficults存入list
                bbox1 = np.expand_dims(b_box_gt,axis=0)
                label_true = np.expand_dims(gt_label,axis=0)
                gt_bboxes += list(bbox1)
                gt_labels += list(label_true)
                break
    
    return pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults

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



if __name__ == '__main__':
    ori_json_file='/home/yuqiushuang/dataset/detection/badcase/old_side_train_result.json' #model1的预测结果
    ori_txt_file = '/home/lichengkun/dataset/TRUNK_highway_20210511_side_coco/infer/train_faster_rcnn/'#model2的预测结果,每一张图片一个txt文件
    #final_result  = '/home/yuqiushuang/dataset/detection/badcase/badcase_labelme_nms/'#错误标注图片存放的路径
    gt_file = '/home/yuqiushuang/dataset/detection/old_train_txt/'
    conf_count = {}

    # #获取yolov4 json格式预测结果和gt相对应的结果
    #pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults = json_convert(ori_json_file,gt_file)
    #获取txt格式预测结果和gt相对应的结果
    pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults = txt_convert(ori_txt_file,gt_file)
#-----------------------------------------------------------------------------------------------------  
#计算每一个种类的置信度的数量
#-----------------------------------------------------------------------------------------------------
    # for scores,labels in zip(pred_scores,pred_labels):
    #     for conf,label in zip(scores,labels):
    #         if str(label) in conf_count :
    #             if 0.8>conf >0.7:
    #                 conf_count[str(label)] += 1
    #             else:
    #                 continue
    #         else:
    #             if 0.8>conf >0.7:
    #                 conf_count[str(label)] = 1
    #             else:
    #                 conf_count[str(label)] = 0
    # print(conf_count) 
    pred_bboxes_count = 0
    for pic in pred_bboxes:
        pred_bboxes_count += len(pic)
    print(pred_bboxes_count)   
#-----------------------------------------------------------------------------------------------------
    #计算precision和recall
#-----------------------------------------------------------------------------------------------------
    # prec,rec,pre_category, rec_category = calc_detection_prec_rec(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults=None,iou_thresh=0.6)

    # print('pre :',pre_category)
    # print('rec :', rec_category)
     
    # #计算AP
    # ap_list = []
    # for rec_every,prec_every in zip(rec,prec):
    #     if rec_every is None or prec_every is None:
    #         ap_list.append(0)
    #         continue
    #     else:
    #         ap = cal_ap(rec_every,prec_every)
    #         ap_list.append(ap)
    # print('ap:',ap_list)
                            


                        
        


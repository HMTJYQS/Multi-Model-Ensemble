from __future__ import print_function

from numpy.core.fromnumeric import shape
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
import xml.dom.minidom
from xml.etree import ElementTree      # 导入ElementTree模块
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from xml.etree.ElementTree import fromstring, ElementTree
from xml.dom.minidom import Document
import glob

# def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行  
#     if element:  # 判断element是否有子元素  
#         if element.text == None or element.text.isspace(): # 如果element的text没有内容  
#             element.text = newline + indent * (level + 1)    
#         else:  
#             element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)  
#     #else:  # 此处两行如果把注释去掉，Element的text也会另起一行  
#     #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level  
#     temp = list(element) # 将elemnt转成list  
#     for subelement in temp:  
#         if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致  
#             subelement.tail = newline + indent * (level + 1)  
#         else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个  
#             subelement.tail = newline + indent * level  
#         prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作  
#     return element
          
# #美化xml
# xml_exm = '/home/yuqiushuang/dataset/detection/badcase/xml_exm/'
# xml_files = os.listdir(xml_exm)
# for xml_file in xml_files:
#     in_file = open(xml_exm+xml_file,'rt')
#     tree = ElementTree.parse(in_file)   # 解析test.xml这个文件，该文件内容如上文  
#     root = tree.getroot()                  # 得到根元素，Element类  
#     root = prettyXml(root, '\t', '\n')     # 执行美化方法  
#     ElementTree.dump(root)                 # 打印美化后的结果
#     tree = ET.ElementTree(root)            # 转换为可保存的结构                
#     xml_save_name = in_file+'_pretty.xml'
#     tree.write(xml_save_name)                 # 保存美化后的结果


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
   #json_files是labelme格式的文件
    #json_files = os.listdir(json_file_path)
   # json_files = sorted(glob.glob(json_file_path+"*.json"))
    total_jpg_files = sorted(glob.glob(img_path+'*.jpg'))
    #print(total_jpg_files[0])
    # print(json_file_path+"*.json")
    #print(total_json_files[0])
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

height = 720
width = 1280
root_dict = {'height':height,'width':width}
xml_file_path = '/home/yuqiushuang/dataset/detection/badcase/company11.xml'
json_file_path = '/home/yuqiushuang/dataset/detection/badcase/labelme/new_00_wbf_cascade_0.95/'
jpg_file_path = '/home/yuqiushuang/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/JPEGImages/'
conf_threshold = 0.98

generate_xml(height,width,conf_threshold,json_file_path,jpg_file_path,xml_file_path)

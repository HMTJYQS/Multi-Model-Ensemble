import os 
import cv2

'''
此脚本用于把gt_txt里面的框画到图片上，并保存到/home/yuqiushuang/dataset/detection/old_train_draw_gt/这个文件夹
'''

gt_txt_file = '/home/yuqiushuang/dataset/detection/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/new_00_txt/'
result_path = '/home/yuqiushuang/dataset/detection/new_00_draw_gt/'
gt_img_file =  '/home/yuqiushuang/dataset/task_trunk_highway_l4_right_normal_bright_20210514_1-2021_05_18_08_21_00-cvatforimages1.1/JPEGImages/'
gt_txt = os.listdir(gt_txt_file)
gt_img = os.listdir(gt_img_file)
for txt in gt_txt:
    file_name =  file_name = txt.split('.')[0] + '.jpg'
    txt_file_path =gt_txt_file + txt
    content = open(txt_file_path,'r')
    for img in gt_img:
        if img == file_name:
            img_file_path =gt_img_file + img #图片img的绝对路径
            img = cv2.imread(img_file_path)
            for line in content.readlines():
                coordinates = line.split()
                cv2.rectangle(img,(int(coordinates[0]),int(coordinates[1])),(int(coordinates[2]),int(coordinates[3])),(0,0,255),2)
                cv2.putText(img,coordinates[6],(int(coordinates[0]),int(coordinates[1])),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # cv2.destroyAllwindows()
            path = result_path+file_name
            cv2.imwrite(path,img)
            break

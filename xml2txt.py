# _*_ coding:utf-8 _*_
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
 

class2id = {'QP':1,"NY":2,"QG":3}
# classes = ["QP", "NY", "QG"] # 根据自己的类别去定义

def convert_annotation(image):
    '''
    给定一张图乡，解析该图像的xml文件，并返回一个字符串，该字符创将写入txt文档的一行

    '''
    image_id = "".join(image.split(".")[:-1])   #test.jpg-->test
    in_file = open('../myData/Annotations/%s.xml'%(image_id),encoding='utf-8')  # 原数据的xml标注文件的路径

    # print(in_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    num_box = 0
    box_info = ""
    for obj in root.iter('object'):
        num_box += 1  # 统计一张图中box的个数
        difficult = obj.find('difficult').text
        cls_ = obj.find('name').text
        if cls_ not in list(class2id.keys()):
            print("没有该label: {}".format(cls_))
            continue
        cls_id = class2id[cls_]
        xmlbox = obj.find('bndbox')

        # b是（x1,x2,y1,y2)---> (x1,y1,w,h)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
        b_w = b[1] - b[0]
        b_h = b[3] - b[2]
        box_this = [str(cls_id),str(b[0]),str(b[2]),str(b_w),str(b_h)]
        box_str = " ".join(box_this)

        box_info += " " + box_str


    image_info = image + " " + str(num_box) + box_info
    # print(image_info)

    return image_info

        

 
if __name__ == "__main__":
    # train
    train_images = os.listdir("./myData/coco/images/train")
    train_txt  = './myData/coco/label_train.txt'

    for image in train_images:
        print("[train] "+image)
        image_info = convert_annotation(image)

        with open(train_txt,'a',encoding='utf-8') as f:
            f.write(image_info+"\n")

    # val
    val_images = os.listdir("./myData/coco/images/val")
    val_txt  = './myData/coco/label_val.txt'

    for image in val_images:
        print("[val] "+image)
        image_info = convert_annotation(image)
        with open(val_txt,'a',encoding='utf-8') as f:
            f.write(image_info+"\n")





# -*- coding: UTF-8 -*-
import cv2
import json
import sys
 
# process bar
def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
 
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
 
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
 


def txt2coco(txt_path="label_train.txt", json_path="train.json",image_path="./myData/coco/image/train/"):

    root_path = image_path # 图像的存储路径
    images, categories, annotations = [], [], []
     
    category_dict = {"QP": 1,"NY":2,"QG":3}
     
    for cat_n in category_dict:
        categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n})
    with open(txt_path, 'r') as f:
        img_id = 0
        anno_id_count = 0
        count = 1
        total = 100
        for line in f.readlines():
            process_bar(count, total)
            count += 1
            line = line.split(' ')
            img_name = line[0]
            bbox_num = int(line[1])
            img_cv2 = cv2.imread(root_path + img_name)
            [height, width, _] = img_cv2.shape
     
            # images info
            images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
     
            """
            annotation info:
            id : anno_id_count
            category_id : category_id
            bbox : bbox
            segmentation : [segment]
            area : area
            iscrowd : 0
            image_id : image_id
            """
            for i in range(0, bbox_num):
                category_id = int(line[i * 5 + 2])
                x1 = float(line[i * 5 + 3])
                y1 = float(line[i * 5 + 4])
                x2 = float(line[i * 5 + 3]) + float(line[i * 5 + 5])
                y2 = float(line[i * 5 + 4]) + float(line[i * 5 + 6])
                width = float(line[i * 5 + 5])
                height = float(line[i * 5 + 6])
     
                bbox = [x1, y1, width, height]
                segment = [x1, y1, x2, y1, x2, y2, x1, y2]
                area = width * height
     
                anno_info = {'id': anno_id_count, 'category_id': category_id, 'bbox': bbox, 'segmentation': [segment],
                             'area': area, 'iscrowd': 0, 'image_id': img_id}
                annotations.append(anno_info)
                anno_id_count += 1
     
            img_id = img_id + 1
     
    all_json = {"images": images, "annotations": annotations, "categories": categories}
     
    with open(json_path, "w") as outfile:
        json.dump(all_json, outfile)



if __name__ == "__main__":
    txt2coco(txt_path="./myData/coco/label_train.txt", json_path="./myData/coco/annotations/boundingbox_train.json",image_path="./myData/coco/images/train/")
    txt2coco(txt_path="./myData/coco/label_val.txt", json_path="./myData/coco/annotations/boundingbox_val.json",image_path="./myData/coco/images/val/")

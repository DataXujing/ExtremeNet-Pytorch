
# VOC分割训练集和测试集
import os
import random
import shutil

trainval_percent = 0.1
train_percent = 0.9
imgfilepath = '../myData/JPEGImages'  #原数据存放地

total_img = os.listdir(imgfilepath)
sample_num = len(total_img)

trains = random.sample(total_img,int(sample_num*train_percent))

for file in total_img:

    if file in trains:
        shutil.copy(os.path.join(imgfilepath,file),"./myData/coco/images/train/"+file)
    else:
        shutil.copy(os.path.join(imgfilepath,file),"./myData/coco/images/val/"+file)

    print(file)
 



 



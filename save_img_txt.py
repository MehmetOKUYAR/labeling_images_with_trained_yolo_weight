import numpy as np
import os
from pathlib import Path
from datetime import datetime

def save_img_txt(image,boxes,class_id,name,save_path):

    height,width,channels = image.shape
    yolo_data =[]
    file_path = os.path.join(save_path,name+ ".txt")
    if len(boxes)>0:
        for cl,bb in zip(class_id,boxes):
            #print('classs id : ',cl)
            #print('box : ',bb)
            x1, y1, w, h = bb[0], bb[1], bb[2], bb[3]
            x_center = (x1+((w-x1)/2))/width
            y_center = (y1+((h-y1)/2))/height
            w = (w-x1)/width
            h = (h-y1)/height
            yolo_data.append([cl,x_center,y_center,w,h])

        
        
        with open(file_path, 'w') as f:
            np.savetxt(
                f,
                yolo_data,
                fmt=["%d","%f","%f","%f","%f"]
            )
    else:
        with open(file_path, 'w') as f:
            np.savetxt(
                f,
                [])

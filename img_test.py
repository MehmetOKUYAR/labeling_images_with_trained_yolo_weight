from curses import meta
import cv2 
import time 
import pycuda.autoinit
from save_img_txt import save_img_txt
from utils.yolo_with_plugins import TrtYOLO
import numpy as np
import logging
import getopt
import sys
import glob
from tqdm import tqdm
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(argv):
    input_file = None
    weight_name = None
    image_type = None
    meta_path = "yolo/obj.names"

    help_str = 'img_test.py -i <input_images_file> -w <trt_weight_name> -t <image_type(jpg,png...)>'

    try:
        opts, args = getopt.getopt(
            argv, "hi:w:t:", ["ifile=","weight_name=","image_type="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)

    # ------------ Parse command line arguments -----------
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-w", "--weight_name"):
            weight_name = arg
        elif opt in ("-t", "--image_type"):
            image_type = arg
    LABELS = []
    with open(meta_path, 'r') as f:
        LABELS = [cname.strip() for cname in f.readlines()]

    # -------- yolov4 tensorrt weight loading --------
    category_num =len(LABELS)
    model_trt= weight_name
    letter_box = False
    trt_yolov4 = TrtYOLO(model_trt,category_num,letter_box)


    def gen_colors(num_colors):
        """Generate different colors.

        # Arguments
        num_colors: total number of colors/classes.

        # Output
        bgrs: a list of (B, G, R) tuples which correspond to each of
                the colors/classes.
        """
        import random
        import colorsys

        hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
        random.seed(1234)
        random.shuffle(hsvs)
        rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
        bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
                for rgb in rgbs]
        return bgrs
    
    COLORS = gen_colors(category_num)

    # --------- yolov4 tensorrt testing function --------
    def YOLOv4(image,name):

        boxes,confidences,classes = trt_yolov4.detect(image,conf_th=0.3)

        # ---------- save prediction results to txt file function -----------
        save_img_txt(image,boxes,classes,name,input_file)
        
        for cl, score, (x_min, y_min, x_max, y_max) in zip(classes, confidences, boxes):
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))

            color = COLORS[int(cl)]
            img = cv2.rectangle(image, start_point, end_point, color, 2)  # draw class box
            label = LABELS[int(cl)]
            text = f'{label}: {score:0.2f}'

            cv2.putText(img, text, (int(x_min), int(y_min-7)), cv2.FONT_ITALIC, 1, COLORS[0], 2)  # print class type with


    images = glob.glob(input_file+'/*.'+image_type)

    # ---------- starting test ----------------
    for i in tqdm(images):
        i.replace('\\','/')
        name = i.split("/")[-1].split(f".{image_type}")[0]

        frame = cv2.imread(i)
        YOLOv4(frame,name)

        cv2.namedWindow('predict_video',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('predict_video',1750,750)
        cv2.imshow("predict_video",frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main(sys.argv[1:])
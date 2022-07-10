# Labeling Images with Trained Yolo Weight
 
 This function I wrote performs the image and txt saving process. You can also use this function on a model you have trained before. If your model works well enough, it will easily detect the images and save the labels. You can edit the saved data and train with your new data quickly.
 
### How to use 
first of all, tensorrt transform using this repository of the yolo weights you have trained 
~~~~~
https://github.com/jkjung-avt/tensorrt_demos
~~~~~~~~~~~
You should edit `yolo/obj.names` according to your project class names

And You should change `plugins` and `utils` your tensorrt_demos file

### Run the application

The input parameters can be changed using the command line :
~~~~~
 python3 img_test.py -i <input_images_file> -w <trt_weight_name> -t <image_type(jpg,png...)>
 ~~~~~~~~~~~~
 
For running with custom weights :
~~~~
python3 img_test.py -i images -w yolov4-tiny -t jpg
~~~~~~~~~

To display the help :
~~~~
 python3 img_test.py -h
~~~~~~

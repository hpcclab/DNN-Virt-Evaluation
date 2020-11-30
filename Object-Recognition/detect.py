import os
import csv
import time
import argparse
from numpy import expand_dims
from tensorflow import device
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from matplotlib import pyplot
#from matplotlib.patches import Rectangle
from helper import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',help='use GPU',action='store_true')
parser.add_argument('-s','--samples', help = 'path to the samples folder',
                    default='./samples/')
parser.add_argument('-o','--output', help = 'path to output folder',
                    default='./output/')
args = parser.parse_args()



# if args.gpu :
#     dev = '/GPU:0'
# else:
#     dev = '/CPU:0'
    
# with device(dev):
input_w, input_h = 416, 416
labels=[]
with open('object_list.txt','r',newline='\n') as infile:
    for label in infile:
         labels.append(label.rstrip('\n'))
 

for sample in os.listdir(args.samples):
    print('\n'+'='*30 + '  SAMPLE#'+sample+'  '+'='*30+'\n')
    sample_folder = args.samples+sample+'/'
    
    timing_header = ['image' , 'size[kb]' , 'preprocessing[s]' ,
              'prediction[s]' ,'postprocessing[s]']
    timing_path = args.output+'output_'+sample+'.csv'
    loadmodel_path = args.output+'loadmodel.csv'          
    timing_out = open(timing_path,'w',newline='')  
    loadmodel_out = open(loadmodel_path ,'w',newline='')
    
    timing_writer = csv.writer(timing_out, delimiter = ',')
    loadmodel_writer = csv.writer(loadmodel_out, delimiter = ',')
    
    timing_writer.writerow(timing_header)
    loadmodel_writer.writerow(['sample#','load_model_time[sec]'])
    
    s0 = time.time()
    model = load_model('model.h5', compile = False)
    e0 = time.time()
    model_time = e0-s0
    loadmodel_writer.writerow([sample,model_time])
    print("The model was loaded in"+
          " {:5.3f} seconds\n".format(model_time))
    loadmodel_writer.writerow([sample, model_time])   
    
    count = 0
    for image_file in os.listdir(sample_folder):
        count +=1
                    
        image_path = sample_folder + image_file            
        
        s1 = time.time()            
        image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))
        e1 = time.time()
        preproc_time = e1-s1
                    
        yhat = model.predict(image)
        e2 = time.time()
        predict_time = e2-e1
                    
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119],
                   [10,13, 16,30, 33,23]]            
        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):            	
        	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold,
                                input_h, input_w)           
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)           
        do_nms(boxes, 0.5) 
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, 
                                                class_threshold)            
        #draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
        e3 = time.time()
        postproc_time = e3-e2            
        
        # summarize what we found
        # for i in range(len(v_boxes)):
        #  	print(v_labels[i], v_scores[i])
        
        timing_writer.writerow([image_file,os.path.getsize(image_path),
                         preproc_time,predict_time,postproc_time])
        if count % 10 ==1:                
            print('\n#      image      ' + '  size[KB]  '+'  pre-proc[sec]  ' +
              '  prediction [sec]  '+ '  post-proc[sec]  ')
            print('-'*82)
        print('{0:<2d} {1:s}   {2:<8.0f}       {3:5.3f}             {4:5.3f}\
              {5:5.3f}'.format(count,image_file,os.path.getsize(image_path),
              preproc_time,predict_time,postproc_time))
              
        
    timing_out.close()
        
loadmodel_out.close()    
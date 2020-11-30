from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from helper import *
import time
import csv
import os
import numpy as np



# load dataset
#data = load('5-celebrity-faces-embeddings.npz')
data = load('../model/lfw-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
s = time.time()
facenet = load_model('../model/facenet_keras.h5')
print('Facenet loaded...')
e=time.time()
print("load time is :"+str(e-s))

ETS=[]
label =[]
directory = '../images/'
for image in listdir(directory):
    ET =[]
    path = directory + image
    print(path)
    for i in range(30):
        print(i)
        start = time.time()
        random_face_pixels = extract_face(path)        
        random_face_emb = get_embedding(facenet,random_face_pixels)        
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        end = time.time()
        ET.append([end-start])
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Execution Time = %3.5f'%(end-start))
    label.append(image)
    ETS.append(ET)

with open('../output/results.csv', 'w',newline='') as resultsfile:
    
    writefile = csv.writer(resultsfile,delimiter=',')    
    writefile.writerow(['Execution Time'])
    
    sNo=0
    for image_ETS in ETS:
        
        writefile.writerow([label[sNo]])
        
        for data in image_ETS:
            print(data)
            writefile.writerow(data)
        sNo +=1
    
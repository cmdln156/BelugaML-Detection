#Adapted from: https://blog.devgenius.io/a-simple-object-detection-app-built-using-streamlit-and-opencv-4365c90f293c

import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time, sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepycopy
import numpy as np
import time
import sys
from os.path import exists

def object_detection_image(config_filename,weights_filename):

    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold =st.sidebar.slider('Confidence Threshold', 0, 100, 50)
        nmsThreshold= st.sidebar.slider('Non-Maximal Suppression Overlap Threshold', 0, 100, 20)
       
        size = 768
        classNames = ['beluga','boat','kayak']
        classColors = [(0,0,255),(255,0,0),(0,255,0)] #Blue = beluga, red = boat, green = kayak
        
        net = cv2.dnn.readNetFromDarknet(config_filename, weights_filename)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs,img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                #Draw box around detected object. Color depends on object type.
                cv2.rectangle(img2, (x, y), (x+w,y+h), classColors[classIds[i]], 2)

                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(img2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColors[classIds[i]], 2)
                
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
             
            st.write(df)
          
        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (size, size), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img2)
    
        st.image(img2, caption='Object Detections')
        
        my_bar.progress(100)



def download_file(file_path):
#Source: https://github.com/streamlit/demo-self-driving

    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()
            
def main():

    st.title('Beluga and Watercraft Detection Using Convolutional Neural Networks')
    st.header('M. Harasyn, W. Chan, E. Ausen, and D. Barber')
    st.subheader('Centre for Earth Observation Science, University of Manitoba')
    
    #Download config file if not present or not the right size.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    object_detection_image("032721.cfg","032721.weights")


# External files to download.
EXTERNAL_DEPENDENCIES = {
    "032721.weights": {
        "url": 'https://github.com/cmdln156/BelugaML-Detection/releases/download/v1.0.0/yolo-obj_032721_best.weights',
        "size": 256059060
    },
    "032721.cfg": {
        "url": 'https://github.com/cmdln156/BelugaML-Detection/releases/download/v1.0.0/yolo-obj_032721.cfg',
        "size": 12931
    }
}
if __name__ == '__main__':
        main()  

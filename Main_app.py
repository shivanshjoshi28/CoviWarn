# Importing all the required libraries
import streamlit as st
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import os.path
import wget
import face_detection 
from tensorflow.keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import time
import streamlit.components.v1 as components  # Import Streamlit
import shutil


# Initialize a Face Detector 
# Confidence Threshold can be Adjusted, Greater values would Detect only Clear Faces
detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# we can also use cascading ... but it doesn't work for longn distance and side view person

# Load Pretrained Face Mask Classfier (Keras Model)
mask_classifier = load_model("model/ResNet50_Classifier.h5")


#################### Streamlit Setup ############################
st.title("Social Distancing Detector")
st.subheader('A GUI Based Social Distancing Monitor System Using Yolo & OpenCV')

cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

MIN_CONF = st.slider(
    'Minimum probability To Filter Weak Detections', 0.0, 1.0, 0.5)
NMS_THRESH = st.slider('Non-Maxima suppression Threshold', 0.0, 1.0, 0.3)

st.subheader('Test Demo Video Or Try Live Detection')
option = st.selectbox('Choose your option',
                      ('Demo1', 'Demo2','Demo3','Demo4' ,'Try Live Detection Using Webcam'))

MIN_CONF = float(MIN_CONF)
NMS_THRESH = float(NMS_THRESH)
USE_GPU = bool(cuda)
MIN_DISTANCE = 50


################# Setup YOLO ####################
weightsPath = "yolo-coco/yolov3.weights"    
if(os.path.isfile(weightsPath)==False):
    file_url = 'https://pjreddie.com/media/files/yolov3.weights'
    file_name = wget.download(file_url)
    weightsPath=file_name


labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
configPath = "yolo-coco/yolov3.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
    st.info("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

###############################################################################################################
if st.button('Start'):
    st.info("[INFO] loading YOLO from disk...")
    st.info("[INFO] accessing video stream...")
    if option == "Demo1":
        vs = cv2.VideoCapture("demo_video/vtest.avi")
    elif option == "Demo2":
        vs = cv2.VideoCapture("demo_video/pedestrians.mp4")
        FILE_PATH="demo_video/pedestrians.mp4"
    elif option == "Demo3":
        vs = cv2.VideoCapture("demo_video/test.mp4")
        FILE_PATH="demo_video/test.mp4"
    elif option == "Demo4":
        vs = cv2.VideoCapture("demo_video/vid_short.mp4")
        FILE_PATH="demo_video/vid_short.mp4"
    else:
        vs = cv2.VideoCapture(0)

    image_placeholder = st.empty()
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    
    ##########save the result###########
    BASE_PATH=""
    path = os.path.join(BASE_PATH, "Results") 
    shutil.rmtree(path)
    # Create Directory for Storing Results (Make sure it doesn't already exists !)
    os.mkdir(BASE_PATH+"Results")
    os.mkdir(BASE_PATH+"Results/Extracted_Faces")
    os.mkdir(BASE_PATH+"Results/Extracted_Persons")
    os.mkdir(BASE_PATH+"Results/Frames")

    n_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)

    #######################################
    for framemain in range(int(n_frames)):
    # while True:

        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))
        height, width, channels = frame.shape
        new_frame_time = time.time()
    
        # Calculating the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = float(fps)
        
        ##############################################################################
        persons = []
        masked_faces = []
        unmasked_faces = []

        # Work on Detected Persons in the Frame
        for boxnum in results:
            x=boxnum[1][0]
            y=boxnum[1][1]
            w=boxnum[1][2]-boxnum[1][0]
            h=boxnum[1][3]-boxnum[1][1]
            persons.append([x,y,w,h])
                
            # Save Image of Cropped Person (If not required, comment the command below)
            cv2.imwrite(BASE_PATH + "Results/Extracted_Persons/"+str(framemain)
                        +"_"+str(len(persons))+".jpg",
                        frame[y:y+h,x:x+w])

            # Detect Face in the Person
            person_rgb = frame[y:y+h,x:x+w,::-1]   # Crop & BGR to RGB
            detections = detector.detect(person_rgb)

            # If a Face is Detected
            if detections.shape[0] > 0:
                detection = np.array(detections[0])
                detection = np.where(detection<0,0,detection)

                # Calculating Co-ordinates of the Detected Face
                x1 = x + int(detection[0])
                x2 = x + int(detection[2])
                y1 = y + int(detection[1])
                y2 = y + int(detection[3])

                try :

                    # Crop & BGR to RGB
                    face_rgb = frame[y1:y2,x1:x2,::-1]   

                    # Preprocess the Image
                    face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)

                    # Predict if the Face is Masked or Not
                    score = mask_classifier.predict(face_arr)

                    # Determine and store Results
                    if score[0][0]<0.5:
                        masked_faces.append([x1,y1,x2,y2])
                    else:
                        unmasked_faces.append([x1,y1,x2,y2])

                    # Save Image of Cropped Face (If not required, comment the command below)
                    cv2.imwrite(BASE_PATH + "Results/Extracted_Faces/"+str(framemain)
                                +"_"+str(len(persons))+".jpg",
                                frame[y1:y2,x1:x2])

                except:
                    continue
        #######################################################################3
        

        violate = set()

        if len(results) >= 2:

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < MIN_DISTANCE:

                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):

            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        
        #####################################################################################
        
        person_count = len(persons)
        safe_count= person_count-len(violate)
        unsafe_count=len(violate)
        masked_face_count = len(masked_faces)
        unmasked_face_count = len(unmasked_faces)
        for f in range(masked_face_count):
    
            a,b,c,d = masked_faces[f]
            cv2.rectangle(frame, (a, b), (c,d), (0,255,0), 2)

        for f in range(unmasked_face_count):

            a,b,c,d = unmasked_faces[f]
            cv2.rectangle(frame, (a, b), (c,d), (0,0,255), 2)

        # Show Monitoring Status in a Black Box at the Top
        cv2.rectangle(frame,(0,0),(width,50),(0,0,0),-1)
        cv2.rectangle(frame,(1,1),(width-1,50),(255,255,255),2)

        xpos = 7

        string = "Total People = "+str(person_count)
        cv2.putText(frame,string,(xpos,35),cv2.FONT_HERSHEY_PLAIN ,1,(255,255,255),1)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_PLAIN ,1,1)[0][0]

        string = " ( "+str(safe_count) + " Safe "
        cv2.putText(frame,string,(xpos,35),cv2.FONT_HERSHEY_PLAIN ,1,(0,255,0),1)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_PLAIN ,1,1)[0][0]

        string = str(unsafe_count)+ " Unsafe ) "
        cv2.putText(frame,string,(xpos,35),cv2.FONT_HERSHEY_PLAIN ,1,(0,0,255),1)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_PLAIN ,1,1)[0][0]
        
        string = "( " +str(masked_face_count)+" Masked "+str(unmasked_face_count)+" Unmasked "+str(person_count-masked_face_count-unmasked_face_count)+" Unknown )"
        cv2.putText(frame,string,(xpos,35),cv2.FONT_HERSHEY_PLAIN ,1,(0,255,255),1)
        cv2.putText(frame, str(fps)+" FPS", (35, height-40), cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 255), 2, cv2.LINE_AA)

        # out_stream.write(frame) # save the output frame int the form of video

        # Save the Frame in frame_no.png format (If not required, comment the command below)
        cv2.imwrite(BASE_PATH+"Results/Frames/"+str(framemain)+".jpg",frame)
        display = 1
        if display > 0:

            image_placeholder.image(
                frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

    
st.success("Design & Developed By Shivansh Joshi")


# Render the h1 block, contained in a frame of size 200x200.
components.html("<html><body>Created by &#10084 <a href='https://github.com/shivanshjoshi28/CoviWarn'> Shivansh Joshi</a></h3></body></html>", width=10000, height=250)

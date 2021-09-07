'''
Calculates Region of Interest(ROI) by receiving points from mouse event and transform prespective so that
we can have top view of scene or ROI. This top view or bird eye view has the property that points are
distributed uniformally horizontally and vertically(scale for horizontal and vertical direction will be
 different). So for bird eye view points are equally distributed, which was not case for normal view.

YOLO V3 is used to detect humans in frame and by calculating bottom center point of bounding boxe around humans, 
we transform those points to bird eye view. And then calculates risk factor by calculating distance between
points and then drawing birds eye view and drawing bounding boxes and distance lines between boxes on frame.
'''

# imports
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import time
import os
import utills, plot # for using helper methods
from datetime import datetime
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import requests
from clint.textui import progress
import streamlit as st
from hello import process
from ex import start
import webbrowser

github = 'https://github.com/shivanshjoshi28/CoviWarn'
website= 'https://nationxbharat.pythonanywhere.com/blog/shivansh'
mouse_pts = []


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click    
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                          
# Below function logic is totally replaced by using the streamlit_drawing_canvas
# def get_mouse_points(event, x, y, flags, param):
#     global mouse_pts
#     # print(x,y)
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(mouse_pts) < 4:
#             cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
#         else:
#             cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
#         if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
#             cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
#             if len(mouse_pts) == 3:
#                 cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
#         if "mouse_pts" not in globals():
#             mouse_pts = []
#         mouse_pts.append((x, y))


def calculate_social_distancing(cap, net,ln1,image_placeholder1,image_placeholder2,image_placeholder3):
    
    count = 0
    vs =cap 

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    # print("Original FPS : ",fps)
    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = utills.get_scale(width, height)

    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    # bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
        
    points = []
    global image
    prev_frame_time=0
    new_frame_time=0
    a=st.empty()
    while True:
        # Calculating the fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = float(fps)

        (grabbed, frame) = vs.read()

        if not grabbed:
            print('Error while loading the video frame')
            break
        (H, W) = frame.shape[:2]
        
        
        # as I have said that this is totally replaced by canvas logic 
        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        # if count == 0:
        #     while True:
        #         image = frame
        #         cv2.imshow("image", image)
                
        #         a.image(
        #             image, caption='Live Social Distancing Monitor Running..!', channels="BGR")
        #         cv2.waitKey(1)
        #         if len(mouse_pts) == 8:
        #             cv2.destroyWindow("image")
        #             a.empty()
        #             break
        points = mouse_pts      
                 
        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
        # This bird eye view then has the property that points are distributed uniformally horizontally and 
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)
        
        ################################## TOP VIEW WRAPPED INPUT FRAME ########################################3

        out = cv2.warpPerspective(frame,prespective_transform,(W,H),flags=cv2.INTER_LINEAR)
        # cv2.imshow("warped",out)
        display2 = 1
        if display2 > 0:
            image_placeholder2.image(
                out, caption=' Wrapped Prespective transform input feed !', channels="BGR")

        ##########################################################################################################
        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]   # finding the correspoding point wrt to the bird eye view


        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
        
        ########################### comment it ###########################
        # print("Distance_w=",distance_w)
        # print("Distance_h=",distance_h)
        
        # pp=np.float32(np.array([points[7:9]]))
        # print("PP = ",pp)
        # ppf=cv2.perspectiveTransform(pp, prespective_transform)[0]
        # print("PPf = ",ppf)
        # print("original ground dist= ",np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2))
        # print("ans = ",utills.cal_dis(ppf[0],ppf[1],distance_w,distance_h))
        # print("second ans= ",utills.cal_dis(warped_pt[0],warped_pt[1],distance_w,distance_h))
        
    ####################################################################################
    
        # YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln1)
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > MIN_CONF:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        # Here we are using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = utills.get_transformed_points(boxes1, prespective_transform)
        
        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        frame1 = np.copy(frame)
        
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
        
        # Show/write image and videos
        if count != 0:
            # output_movie.write(img)
            # bird_movie.write(bird_image)
            cv2.putText(img, str(fps)+" FPS", (35, H-40), cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 255), 2, cv2.LINE_AA)
    
            # cv2.imshow('Bird Eye View', bird_image)
            # cv2.imshow("New image",img)
            display4=1
            if display4 > 0:
                image_placeholder1.image(
                    img, caption='Live Social Distancing Monitor Running..!', channels="BGR")

            display3 = 1
            if display3 > 0:
                image_placeholder3.image(
                    bird_image, caption='Bird Eye View', channels="BGR")

            # cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            # cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)
    
        count = count + 1
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
     
    vs.release()
    # cv2.destroyAllWindows() 
        


# this function is taken from the www.discuss.streamlit.com 
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield


def formatFloat(num):
    return "{:.2f} ".format(num)

# this is to show the downloading process on the screen 
def DownloadWeightFile():
    file_url = 'https://pjreddie.com/media/files/yolov3.weights'
    r = requests.get(file_url, stream=True)
    path = weightsPath
    count = 0
    count_tmp = 0
    with st_capture(output.code):
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            time1 = time.time()
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()
                    count += len(chunk)
                    if time.time() - time1 > 2:
                        percentage = count / total_length * 100
                        speed = (count - count_tmp) / 1024 / 1024 / 2
                        count_tmp = count
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        strdetail=dt_string+" "+formatFloat(percentage) + '%   Completed ' + 'Speed: '+ formatFloat(speed) + 'M/S '
                        print(strdetail)
                        time1 = time.time()
        print("100 % Completed Congrats .. Now You can start CoviWarn")



# this function will simply get the first frame of the video and save it to the server side directory
def processFrame(vs):
    vidcap = vs
    success,image = vidcap.read()
    got=False
    try:
        while got==False:
            cv2.imwrite("frame.jpg",image)     # save frame as JPEG file      
            success,image = vidcap.read()
            # print('Read a new frame: ', success)
            got=True
    except:
        print("wait rendering error !!")
MIN_CONF=0.5
NMS_THRESH=0.3
USE_GPU=True
model_path="models/"
weightsPath = model_path+"yolov3.weights"   
output=st.empty()



def app():

    #################### Streamlit Setup ############################
    st.title("Social Distancing Detector🦠")
    components.html('<iframe src="https://giphy.com/embed/hpXFi66bfQm7e81ohw" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>',width=480,height=480)
    st.subheader('A GUI Based Social Distancing Monitor System Using Yolo & OpenCV')
    global MIN_CONF
    global NMS_THRESH
    global USE_GPU
    global model_path
    global weightsPath
    global output
    global mouse_pts

    cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

    MIN_CONF = st.slider(
        'Minimum probability To Filter Weak Detections', 0.0, 1.0, 0.5)
    NMS_THRESH = st.slider('Non-Maxima suppression Threshold', 0.0, 1.0, 0.3)

    st.subheader('Test Demo Video Or Try Live Detection')
    option = st.selectbox('Choose your option',
                        ('Sample_Video_1', 'Sample_Video_2','Sample_Video_3','Sample_Video_4','Sample_Video_5' ,'Try Live Detection Using Webcam'))
    warn=st.empty()
    warn.warning(f"You can't change the above Dropbox afterward. Please choose wisely.ThankYou !!")
    MIN_CONF = float(MIN_CONF)
    NMS_THRESH = float(NMS_THRESH)
    USE_GPU = bool(cuda)


    





    model_path="models/"
    weightsPath = model_path+"yolov3.weights"   
    output = st.empty()
    # load Yolov3 weights
    if(os.path.isfile(weightsPath)==False):
        DownloadWeightFile()
        
    configPath = model_path + "yolov3.cfg"
    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    if USE_GPU:
        st.info("[INFO] setting preferable backend and target to CUDA...")
        net_yl.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net_yl.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    n_length,mouse_pts,opencvimage,original_pts_len=process()
    # print(mouse_pts)
    Bug=False
    if(n_length!=None and original_pts_len!=None and n_length!=original_pts_len):
        Bug=True
        st.warning("You are marking too fast or exceeded the limit 🥺. Clear (🗑️) the canvas Again ")

    if(Bug==False and n_length!=None and n_length<7):
        rem=7-n_length
        
        st.info(f"You need to mark {rem} points more 😄 !!")
        
        
    if(Bug==False and n_length!=None and n_length>7):
        st.warning("You have marked {n_length} points 🥺! Please mark 7 points Only. Now clear the Canvas above :) ")


    # below is to show the canvas , in which we will mark out point 
    if np.all(opencvimage) is not None:
        st.image(opencvimage, caption='ROI preview', channels="BGR")# showing the Border point on the streamlit server


    button1=st.button('Download First Frame ')
    if(button1==True):
        print(option)
        warn.warning(f"Don't change above :) ")
        FILE_PATH="demo_video/CCTV_demo.mp4"
        if option == "Sample_Video_1":
            vs = cv2.VideoCapture("demo_video/CCTV_demo.mp4")
        elif option == "Sample_Video_2":
            vs = cv2.VideoCapture("demo_video/pedestrians.mp4")
            FILE_PATH="demo_video/pedestrians.mp4"
        elif option == "Sample_Video_3":
            vs = cv2.VideoCapture("demo_video/test.mp4")
            FILE_PATH="demo_video/test.mp4"
        elif option == "Sample_Video_4":
            vs = cv2.VideoCapture("demo_video/vid_short.mp4")
            FILE_PATH="demo_video/vid_short.mp4"
        elif option == "Sample_Video_5":
            vs = cv2.VideoCapture("demo_video/vtest.avi")
            FILE_PATH="demo_video/vtest.avi"
        else:
            vs = cv2.VideoCapture(0)
        processFrame(vs)
        st.info("Frame Loaded successfully")
        st.markdown(start(), unsafe_allow_html=True)


    # if st.button('Start'):
    if(n_length!=None and n_length==7):
        warn.warning(f"Don't change above :) ")
        st.info("[INFO] loading YOLO from disk...")
        st.info("[INFO] accessing video stream...")
        if option == "Sample_Video_1":
            vs = cv2.VideoCapture("demo_video/CCTV_demo.mp4")
        elif option == "Sample_Video_2":
            vs = cv2.VideoCapture("demo_video/pedestrians.mp4")
            FILE_PATH="demo_video/pedestrians.mp4"
        elif option == "Sample_Video_3":
            vs = cv2.VideoCapture("demo_video/test.mp4")
            FILE_PATH="demo_video/test.mp4"
        elif option == "Sample_Video_4":
            vs = cv2.VideoCapture("demo_video/vid_short.mp4")
            FILE_PATH="demo_video/vid_short.mp4"
        elif option == "Sample_Video_5":
            vs = cv2.VideoCapture("demo_video/vtest.avi")
            FILE_PATH="demo_video/vid_short.mp4"
        else:
            vs = cv2.VideoCapture(0)
        # print("\n")
        # print("Final mouse points---",mouse_pts)
            
        image_placeholder1 = st.empty()
        image_placeholder2=st.empty()
        image_placeholder3=st.empty()
        calculate_social_distancing(vs, net_yl, ln1,image_placeholder1,image_placeholder2,image_placeholder3)
    st.success("Design & Developed By Shivansh Joshi 👨‍💻")
    if st.button('View Code ❤️ Github'):
        webbrowser.open_new_tab(github)
    if st.button('Visit my Profile'):
        webbrowser.open_new_tab(website)
    components.html("<html><body><h3>©️ coviwarn by Shivansh Joshi</h3></body></html>", width=10000, height=250)





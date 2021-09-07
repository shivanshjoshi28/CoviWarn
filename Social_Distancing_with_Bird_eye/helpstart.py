import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import webbrowser

github = 'https://github.com/shivanshjoshi28/CoviWarn'
website= 'https://nationxbharat.pythonanywhere.com/blog/shivansh'

def app():
    st.title("Social Distancing Detector🦠 ")
    st.header('Please follow the steps mentioned below😂')
    components.html('<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="https://giphy.com/embed/dlMIwDQAxXn1K" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div>',width=480,height=360)
    
    st.subheader("Step 1 : ")
    st.write("1️⃣ Set the slider")
    st.write("2️⃣ Select demo video of your choice, on which you wanna apply the Social distance algorithm")
    st.warning("Keep the note that you can't choose other video from the dropdown after this step")
    First=cv2.imread("Instruction/First.png")
    st.image(First,caption='Step 1', channels="BGR")

    st.subheader("Step 2 : ")
    st.write("1️⃣ Next click Download First Frame Button")
    Second=cv2.imread("Instruction/Second.png")
    st.image(Second,caption='Step 2', channels="BGR")
    
    st.subheader("Step 3 : ")
    st.write("1️⃣ Upload the file that we have downloaded previously before this step.")
    st.write("2️⃣ Wait for the time the app is showing RUNNING on the top right")
    Third=cv2.imread("Instruction/Third.png")
    st.image(Third,caption='Step 3', channels="BGR")
    
    st.subheader("Step 4 : ")
    st.write("1️⃣ Slowly and calmy mark 4 corner points for the ROI")
    st.write("2️⃣ Now mark three position on image that will be finalizing our criteria for social distancing")
    st.write("3️⃣ After marking total of 7 points you are ready with your answers ")
    st.info("[info] You can read about this in much detail . Scroll Below !")
    Fourth=cv2.imread("Instruction/Fourth.png")
    st.image(Fourth,caption='Step 4', channels="BGR")
    st.success("Hope You like this !! ThankYou ")
    
    
    st.header("Hey Developers !! Are you getting some errors or bug")
    st.subheader("Wait a minute . Calm Down .")   
    st.write("Either restart the app or click the 🗑️ button on canvas") 
    Fifth=cv2.imread("Instruction/Fifth.png")
    st.image(Fifth,caption='Resolve Bug/Errors', channels="BGR")
    st.markdown('''
                # About This Project
   This is basically a tool to monitor social distancing from CCTV, videos using Python, Deep learning, Computer Vision. This tool can 
   automatically estimate interpersonal distance from uncalibrated RGB cameras. Can be used at public places and workplace.

   In the fight against the COVID-19, social distancing has proven to be a very effective measure. To ensure social
   distancing protocol in public places and workplace, I have developed social distancing detection tool that can monitor
   if people are keeping a safe distance from each other by analyzing real time video streams from the camera.

   This app has following features:

   * Detect humans in the frame with yolov3.
   * Calculates the distance between every human who is detected in the frame.
   * Shows how many people are at High, Low and Not at risk.
                
                ''')
    
    

    st.header("Objection Detection through Yolo ( Pretrained model )🧑‍💻 ")
    YOLO=cv2.imread("Instruction/yolo_design.jpg")
    st.image(YOLO,caption='Yolo Working', channels="BGR")
    st.write("Compared to other region proposal classification networks (fast RCNN) which perform detection on various region proposals and thus end up performing prediction multiple times for various regions in a image, Yolo architecture is more like FCNN (fully convolutional neural network) and passes the image (nxn) once through the FCNN and output is (mxm) prediction. This the architecture is splitting the input image in mxm grid and for each grid generation 2 bounding boxes and class probabilities for those bounding boxes. Note that bounding box is more likely to be larger than the grid itself.")
    Seventh=cv2.imread("Instruction/Seventh.png")
    st.image(Seventh,caption='After applying the model', channels="BGR")
    
    st.header("Calculating distance using calibration through ROI and Bird Eye view")
    Sixth=cv2.imread("Instruction/Sixth.png")
    st.image(Sixth,caption='', channels="BGR")
    st.latex(r'''
             \text { Ground } x=\frac{180 * \text { newpixelx }}{x \text { (Figure })}
             ''')
    st.latex(r'''\text { Groundy }=\frac{180 * \text { newpixely }}{y(\text { Figure })}''')
    st.subheader("Some important terms used above")
    st.write("1️⃣ Here 'newpixelx' is the aerial distance calculated in pixel in the x direction")
    st.write("2️⃣ Here 'newpixely' is the aerial distance calculated in pixel in the y direction")
    st.write("3️⃣ 'Groundx' is the original ground distance between two point in x direction ")
    st.write("4️⃣ 'Groundy' is the original ground distance between two point in y direction ")
    st.write("5️⃣ x and y direction is defined by the user while giving first four point as 1-2 point line indicates y direction and 2-3 point line indicated x direction")
    st.success("Design & Developed By Shivansh Joshi 👨‍💻")
    if st.button('View Code ❤️ Github'):
        webbrowser.open_new_tab(github)
    if st.button('Visit my Profile'):
        webbrowser.open_new_tab(website)
    components.html("<html><body><h3> ©️ coviwarn by Shivansh Joshi</h3></body></html>", width=10000, height=250)
    
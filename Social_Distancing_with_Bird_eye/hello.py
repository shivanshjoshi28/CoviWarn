import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
mouse_pts = []
# Specify canvas parameters in application
# function help taken from https://github.com/andfanilo/streamlit-drawable-canvas. Big Thank to creator of this repo 
def process():
    global mouse_pts
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg","mp4"])
    drawing_mode = "circle"
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if canvas_result.json_data is not None:
        st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
        n_lines = len(canvas_result.json_data["objects"])
        open_cv_image=None
        
        # print("n lines",n_lines)
        if(n_lines==1):
                # print("changing")
                # if "mouse_pts" not in globals():
                mouse_pts = []
        if(n_lines>0 and n_lines<8):
            dic = canvas_result.json_data["objects"][-1]
            y=int(dic['top'])
            x=int(dic['left'])
            # print("Bg-Images",bg_image)
            if( bg_image is not None):
                pil_image=Image.open(bg_image).convert('RGB')
                open_cv_image = np.array(pil_image) 
                # Convert RGB to BGR 
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                # print(open_cv_image.shape)
                x=int(x/600*open_cv_image.shape[1])
                y=int(y/400*open_cv_image.shape[0])
                # st.write(str(x))
                # st.write(str(y))
                # print("X= ",x)
                # print("Y= ",y)
                
                # print("len of mouse point",len(mouse_pts))
                if(len(mouse_pts)>=1):
                    # print("Hello")
                    open_cv_image=cv2.imread("new.jpg")
                if len(mouse_pts) < 4:
                    cv2.circle(open_cv_image, (x, y), 5, (0, 0, 255), 10)
                else:
                    cv2.circle(open_cv_image, (x, y), 5, (255, 0, 0), 10)
                    
                if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
                    cv2.line(open_cv_image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
                    if len(mouse_pts) == 3:
                        cv2.line(open_cv_image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
                if "mouse_pts" not in globals():
                    mouse_pts = []
                mouse_pts.append((x,y))
                # while True:
                #     cv2.imshow("hello",open_cv_image)
                # sleep(10)
                cv2.imwrite("new.jpg",open_cv_image)
            # print(n_lines)
        if(n_lines>=7):
            # print("Returned")
            aa=mouse_pts
            mouse_pts=[]
            return n_lines,aa,open_cv_image,len(aa)
        else:
            return n_lines,[],open_cv_image,len(mouse_pts)
    return None,[],None,None

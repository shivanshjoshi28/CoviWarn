# CoviWarn - Social Distancing Detector ( GUI )
## ROI and Bird eye view features for calculating distance 
   Tool to monitor social distancing from CCTV, videos using Python, Deep learning, Computer Vision. This tool can 
   automatically estimate interpersonal distance from uncalibrated RGB cameras. Can be used at public places and workplace.

   In the fight against the COVID-19, social distancing has proven to be a very effective measure. To ensure social
   distancing protocol in public places and workplace, I have developed social distancing detection tool that can monitor
   if people are keeping a safe distance from each other by analyzing real time video streams from the camera.

   This tool has following features:

   * Detect humans in the frame with yolov3.
   * Calculates the distance between every human who is detected in the frame.
   * Shows how many people are at High, Low and Not at risk.
   

   ## Demo:
   https://user-images.githubusercontent.com/58811384/132108463-03cab3ec-6cdb-424a-8608-09a89cc1c0ad.mp4


## Requirements:

    You will need the following to run this code:
    Python 3
    Opencv(CV2) 4.x
    numpy
    
    For human detection:
    yolov3.weights, yolov3.cfg files (weights file in not present because of size issue. It can be downloaded from 
    here : https://pjreddie.com/media/files/yolov3.weights ...... It will be automatically download it, if not present)
    
    For running: 
    Good GPU, for faster results. CPU is also fine(I have tried it on CPU).
    
## Usage:
     Run the command below after coming to this directory
     *  pip install -r requirements.txt
     *  streamlit run App.py
            

## Idea Credits:

   [Landing.ai](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/)
   
## References:

   Yolov3 object detection : https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
   Prespective Transform : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
   
   Article :[Social Distancing AI](https://medium.com/@birla.deepak26/social-distancing-ai-using-python-deep-learning-c26b20c9aa4c)

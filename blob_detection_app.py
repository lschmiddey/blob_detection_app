import streamlit as st
import cv2
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd

st.write("""
# Simple Blob Detection App
Upload your image and see where the Blob is!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    minCircularity = st.sidebar.slider('minCircularity', 0., 1., 0.8)
    minArea = st.sidebar.slider('minArea', 0, 1, 10)
    maxArea = st.sidebar.slider('maxArea', 0, 1, 100000)
    
    data = {'minCircularity': minCircularity,
            'minArea': minArea,
            'maxArea': maxArea}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByInertia = False
blob_params.filterByConvexity = False
blob_params.filterByColor = True
blob_params.blobColor = 0
blob_params.filterByCircularity = True
blob_params.filterByArea = False
blob_params.minCircularity = df.minCircularity.values.item()
blob_params.minArea = df.minArea.values.item()
blob_params.maxArea = df.maxArea.values.item()
blob_detector = cv2.SimpleBlobDetector_create(blob_params)


st.subheader('User Input parameters')
st.write(df)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:   
    # Read in and make greyscale
    PILim = Image.open(uploaded_file).convert('L')
    # Make Numpy/OpenCV-compatible version
    openCVim = np.array(PILim)
    openCVim = cv2.bitwise_not(openCVim)
    st.image(openCVim, caption='Uploaded Image.', use_column_width=True)
    
    keypoints = blob_detector.detect(openCVim)

    # find largest blob
    if len(keypoints) > 0:
        kp_max = keypoints[0]
        for kp in keypoints:
            if kp.size > kp_max.size:
                kp_max = kp
                
    pts = np.array([kp_max.pt])
    
    data_coordinates = {'x_coordinate': int(pts[:, 0]),
            'y_coordinate': int(pts[:, 1])}    
    df_coordinates = pd.DataFrame(data_coordinates, index=[0])   

    im_with_keypoints = cv2.cvtColor(openCVim,cv2.COLOR_GRAY2RGB)
#     im_with_keypoints = cv2.circle(openCVim, (int(pts[:, 0]), int(pts[:, 1])), 50, color=(0,255,0), thickness=30, lineType=8, shift=0)
    im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, [kp_max], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)


    st.image(im_with_keypoints, caption='Image with Blob.', use_column_width=True)
    
    st.write(df_coordinates)
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import cv2
import csv
import os
import pandas as pd
import base64
from IPython.display import HTML

# Lets try to integrate streamlit and mediapipe

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic

DEMO_VIDEO = 'D:\\Face-Mesh-MediaPipe\\Demos\\bella porch to demo.mp4'

st.title('Movement Detection Application using MediaPipe')


st.sidebar.title('Movement Detection Application using MediaPipe')
st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    b64 = base64.b64encode(df.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Run on Video', 'Dataset']
                                )

if app_mode == 'About App':

    st.markdown(
        'In this application we are using **MediaPipe** from Google for creating a Movement Detection on a video')

    st.text('**Djagora Team**')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-chisld {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image(
        'https://scontent.ftun7-1.fna.fbcdn.net/v/t1.6435-9/117716527_305096250909701_5175488458046493683_n.png?_nc_cat=106&ccb=1-5&_nc_sid=e3f864&_nc_ohc=qozk54OdaM8AX8Go6fq&tn=Lf16wLVU8l36_cSv&_nc_ht=scontent.ftun7-1.fna&oh=6f73b59a15606ad45aef37b1dc7f3a4d&oe=61442316')

elif app_mode == 'Run on Video':
     
    st.subheader('We are applying Holonitic on a video')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)
        st.sidebar.text('Params For video')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    st.markdown(' ## Output')
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)

    st.sidebar.text('Input Video')
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_holistic.Holistic(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True

            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)

            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )
            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark  # extraire tt les pts repere (pose)
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
                                          pose]).flatten())  # extraire les diff coordonnées ds un tab numpy

                # Extract Face landmarks
                face = results.face_landmarks.landmark  # extraire tt les pts repere (face)
                face_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                # face_row: maw face fiha x1  y1 z1 v1 kol wahda wahadha , fel face_row yet7Ato fard vect [x1 y1 z1 v1],[x2 y2 z2 v2]
                # Concate rows
                row = pose_row + face_row

                # Append class name
                # row.insert(0, class_name)
                # Export to CSV
                with open('C:\\Users\\MSI\\Desktop\\Face-Mesh-MediaPipe\\Demos\\coordonnees.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                    
            except:
                pass
            
            image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            image = image_resize(image=image, width=640)
            stframe.image(image, channels='BGR', use_column_width=True)

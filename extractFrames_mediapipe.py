'''
	Shofiyati Nur Karimah
    January 2023

    Etracting Dataset using Mediapipe Holistic
    Output files = extracted feature from Mediapipie Holistic saved in CSV format
    [Extract 1] = output files are saved in specific folders (Test/Train/Validation)
    [Extract 2] = output files saved in the same directory as this file
'''

import os
import cv2
import mediapipe as mp
import numpy as np
import csv


# dataset = os.listdir('DataSet_DAiSEE/') # [Extract 1]
dataset = "../../Dataset/DAiSEE/DataSet_DAiSEE/" #[Extract 2]
if '.DS_Store' in dataset:
    dataset.remove('.DS_Store')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# extracting frames
# def get_frame(video_file,destination_path, set): #[Extract 1]
def get_frame(video_file,destination_path): #[Extract 2]
    filename_format = "{:s}.{:s}"
    ext = "csv"
    filename = filename_format.format(video_file[:-4],ext)

    with open('header.csv','r') as file:
        header = csv.reader(file)
        header = next(header)

    # with open(set+filename,"w", newline='') as csvfile: #[Extract 1]
    with open(filename,"w", newline='') as csvfile: #[Extract 1]
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

    cap = cv2.VideoCapture(destination_path) 
    ret_val = True
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        while True:
            ret, image = cap.read()

            if not ret:
                ret_val = False
                break
            
            # Recolor feed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make detections
            results = holistic.process(image)

            # create header row for csv
            # Run the following lines only once to create the dataset header
            # =========================================================================================
            # num_coords = len(results.face_landmarks.landmark)+len(results.pose_landmarks.landmark)
            # row =[]
            # for val in range(1,num_coords+1):
            #     row += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            # with open(sets+filename, mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)
            # =========================================================================================

            # # pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks
            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.face_landmarks, 
                mp_holistic.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1))

            # 2. Right hand
            mp_drawing.draw_landmarks(
                image=image, 
                landmark_list=results.right_hand_landmarks, 
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

            # 3. Left Hand
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

            # 4. Pose Detector
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=2))

            ## Export coordinates
            try:
                # Extract pose landmark
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                        for landmark in pose]).flatten())
                
                # Extract face landmark
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                        for landmark in face]).flatten())
                
                # Concatenate rows
                row = pose_row+face_row
                               
                # Export to CSV
                # with open(set+filename, mode='a', newline='') as f: #[Extract 1]
                with open(filename, mode='a', newline='') as f: #[Extract 1]
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            
            except:
                pass

            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                ret_val = False
                break
            
        cap.release()
        cv2.destroyAllWindows()
        return ret_val

# [Extract 1]
# ==========================================================================================
# for ttv in dataset:
#     if not ttv.startswith('.'):
#         if not ttv.endswith('*.csv'):
#             users = os.listdir('DataSet_DAiSEE/'+ttv+'/')
#             path_sets = os.path.abspath('.')+'/DataSet_DAiSEE/'+ttv+'/'
#             # print("Path sets{}".format(path_sets))
#             for user in users:
#                 if not user.startswith('.'):
#                     currUser = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/')
#                     for extract in currUser:
#                         if not extract.startswith('.'):
#                             clip = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/')[0]
#                             print ("Processed video {}".format(clip[:-4]))
#                             path = os.path.abspath('.')+'/DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/'
#                             get_frame(clip,path,path_sets)

# [Extract 2]
# ===========================================================================================
for subdirs, dirs, files in os.walk(dataset):
    if '.DS_Store' in (subdirs, dirs, files):
        dataset.remove('.DS_Store')

    for clip in files:
        # print(path_set)
        if clip.endswith(".avi"):
            path = os.path.join(subdirs,clip)
            print(path)
            get_frame(clip,path)
        if clip.endswith(".mp4"):
            path = os.path.join(subdirs,clip)
            print(path)
            get_frame(clip,path)
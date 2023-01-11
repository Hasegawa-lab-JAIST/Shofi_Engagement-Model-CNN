'''
	Shofiyati Nur Karimah
    January 2023

    Etracting Dataset in sub folders using Mediapipe
'''

import os
import cv2
import mediapipe as mp
import numpy as np
import csv


dataset = os.listdir('DataSet_DAiSEE/')
if '.DS_Store' in dataset:
    dataset.remove('.DS_Store')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # extracting frames
# def get_frame(video_file,destination_path):
#     filename_format = "{:s}.{:s}"
#     ext = "csv"
#     filename = filename_format.format(video_file,ext)

#     # with open(filename,"w") as f:
#     #     f.write()

#     cap = cv2.VideoCapture(destination_path+video_file)
#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#             _, image = cap.read()

#             # Recolor feed
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # Make detections
#             results = holistic.process(image)
#             print(results)
#             # print(results.face_landmarks)
#             # print(results.pose_landmarks)

#             num_coords = len(results.face_landmarks.landmark)+len(results.pose_landmarks.landmark)

#             # create header row for csv
#             # Run the following lines only once to create the dataset header
#             # =========================================================================================
#             for val in range(1,num_coords+1):
#                 ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
#             with open(filename, mode='w', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 # csv_writer.writerow(row)
#             # =========================================================================================

#             # # pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks
#             # Recolor image back to BGR for rendering
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             # 1. Draw face landmarks
#             mp_drawing.draw_landmarks(
#                 image, 
#                 results.face_landmarks, 
#                 mp_holistic.FACE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1),
#                 mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1))

#             # 2. Right hand
#             mp_drawing.draw_landmarks(
#                 image=image, 
#                 landmark_list=results.right_hand_landmarks, 
#                 connections=mp_holistic.HAND_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

#             # 3. Left Hand
#             mp_drawing.draw_landmarks(
#                 image, 
#                 results.left_hand_landmarks, 
#                 mp_holistic.HAND_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
#                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

#             # 4. Pose Detector
#             mp_drawing.draw_landmarks(
#                 image, 
#                 results.pose_landmarks, 
#                 mp_holistic.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=3),
#                 mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=2))

#             ## Export coordinates
#             try:
#                 # Extract pose landmark
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
#                                         for landmark in pose]).flatten())
                
#                 # Extract face landmark
#                 face = results.face_landmarks.landmark
#                 face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
#                                         for landmark in face]).flatten())
                
#                 # Concatenate rows
#                 row = pose_row+face_row
                
#                 # # Append class name
#                 # row.insert(0, class_name)
                
#                 # Export to CSV
#                 with open(filename, mode='a', newline='') as f:
#                     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                     csv_writer.writerow(row)
            
#             except:
#                 pass

#             cv2.imshow('Raw Webcam Feed', image)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

for ttv in dataset:
    if not ttv.startswith('.'):
        users = os.listdir('DataSet_DAiSEE/'+ttv+'/')
        for user in users:
            if not user.startswith('.'):
                currUser = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/')
                for extract in currUser:
                    if not extract.startswith('.'):
                        clip = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/')[0]
                        print (clip[:-4])
                        path = os.path.abspath('.')+'/DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/'
                        # get_frame(clip,path)
                        filename_format = "{:s}.{:s}"
                        ext = "csv"
                        filename = filename_format.format(clip,ext)

                        # with open(filename,"w") as f:
                        #     f.write()

                        cap = cv2.VideoCapture(path+clip)
                        with mp_holistic.Holistic(
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as holistic:
                            while cap.isOpened():
                                _, image = cap.read()

                                # Recolor feed
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                # Make detections
                                results = holistic.process(image)
                                # print(results.face_landmarks)
                                # print(results.pose_landmarks)

                                num_coords = len(results.face_landmarks.landmark)+len(results.pose_landmarks.landmark)
                                # create header row for csv
                                # Run the following lines only once to create the dataset header
                                # =========================================================================================
                                for val in range(1,num_coords+1):
                                    ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                                with open(filename, mode='w', newline='') as f:
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    # csv_writer.writerow(row)
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
                                    
                                    # # Append class name
                                    # row.insert(0, class_name)
                                    
                                    # Export to CSV
                                    with open(filename, mode='a', newline='') as f:
                                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                        csv_writer.writerow(row)
                                
                                except:
                                    pass

                                cv2.imshow('Raw Webcam Feed', image)
                                if cv2.waitKey(5) & 0xFF == ord('q'):
                                    break

                        cap.release()
                        cv2.destroyAllWindows()



# # Define input video path
# input_video = '/Users/hasegawa-lab/OneDrive - Japan Advanced Institute of Science and Technology/Documents/Dataset/DAiSEE/DataSet_DAiSEE/Test/500044/5000441001.avi'

# cap = cv2.VideoCapture(input_video)

# fps = cap.get(cv2.CAP_PROP_FPS)

# # output image
# output_dir = '/Users/hasegawa-lab/OneDrive - Japan Advanced Institute of Science and Technology/Documents/Dataset/DAiSEE/'

# # initialize a frame counter
# frame_count = 0

# # Process the video frames
# while True:
#     ret, frame = cap.read() 

#     # break the loop if the video has ended
#     if not ret:
#         break

#     # output path
#     output_path = output_dir + 'frame' + str(frame_count) + '.jpg'
#     cv2.imwrite(output_path, frame)

#     # increment the frame counter
#     frame_count += 1

#     # print current number and the total number of frames
#     print('Processing frame:', frame_count, '/', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

#     cap.release()

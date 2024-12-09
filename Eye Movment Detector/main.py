import math
import cv2
import mediapipe as mp
import time
import numpy as np
import socket
# for sound
import pygame
from pygame import mixer  # mixer allow us to play sound available in directory

# variable
frame_counter=0
CEF_counter=0  # CEF- Closed Eye Frame
Total_blinks=0

#for voice
start_voice=False
counter_left=0
counter_right=0

#constant
FONTS=cv2.FONT_HERSHEY_COMPLEX
CLOSED_EYES_FRAME=2

# for voice- initializing mixer
mixer.init()

# load voices
voice_outside=mixer.Sound('Outside_Beep.wav')

# put the indexes of the landmarks for different components on face
# Left eyes indices
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#for socket
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1",5052)

# for setting up the mediapipe
map_face_mesh=mp.solutions.face_mesh

# videoCapture-for input
camera=cv2.VideoCapture(0)

# function for landmark detection
def landmarkDetection(img,results,draw=False):

    img_height,img_width=img.shape[:2]  # take 1st two parameter from shape ignore 3rd para-no of channel
    mesh_coords=[(int(point.x*img_width), int(point.y*img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img,p,2,(0,255,0),-1) for p in mesh_coords]
    return mesh_coords

def eyeExtractor(img,right_eye_coords,left_eye_coords):

    # converting to gray scale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #dimesions of image
    dim=gray.shape

    #creating mask from gray scale dim
    mask=np.zeros(dim,dtype=np.int8)

    #drawing eyes shape on mask with white color
    cv2.fillPoly(mask,[np.array(right_eye_coords,dtype=np.int32)],255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    #show the mask
    cv2.imshow('mask',mask)

    #draw eyes on image of mask , where white shape present
    eyes=cv2.bitwise_and(gray,gray,mask=mask)

    cv2.imshow('draw eyes', eyes)
    #change black color to gray,other than eyes- bcz when we count pixels that time this black
    #color affects
    eyes[mask==0]=155

    # for separating eyes,we have to draw rectangle for that we are taking min and max, value of x and y position for both the eyes

    r_max_x=(max(right_eye_coords,key=lambda item: item[0]))[0]
    r_min_x=(min(right_eye_coords,key=lambda item: item[0]))[0]
    r_max_y=(max(right_eye_coords,key=lambda item: item[1]))[1]
    r_min_y=(min(right_eye_coords,key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # cropping eye from the mask
    cropped_right=eyes[r_min_y:r_max_y,r_min_x:r_max_x]
    cropped_left=eyes[l_min_y:l_max_y,l_min_x:l_max_x]

    #returning the cropped eyes
    return cropped_right,cropped_left

def positionEstimator(cropped_eye):

    #getting height and width of eye
    h,w=cropped_eye.shape

    # remove the noise from image
    guassian_blur=cv2.GaussianBlur(cropped_eye,(5,5),0)
    median_blur=cv2.medianBlur(cropped_eye,3)

    # applying thresholding to convert binary_image
    ret,threshold_eye=cv2.threshold(median_blur,45,255,cv2.THRESH_BINARY)

    # create fixed part for eye width
    piece=int(w/3)

    # taking about eyes into three parts
    right_piece=threshold_eye[0:h,0:piece]
    center_piece=threshold_eye[0:h,piece:piece+piece]
    left_piece=threshold_eye[0:h,piece+piece:w]

    # calling pixel counter function
    eye_position,color=pixelCounter(right_piece,center_piece,left_piece)

    return eye_position,color
def pixelCounter(first_piece,second_piece,third_piece):

    #counting each pixel in each part
    right_part=np.sum(first_piece==0)
    center_part=np.sum(second_piece==0)
    left_part=np.sum(third_piece==0)

    #creating list of these values
    eye_parts=[right_part,center_part,left_part]

    #getting the index of max values in the list
    max_index=eye_parts.index(max(eye_parts))

    # for sending data to unity
    data=str.encode(str(max_index))
    sock.sendto(data,serverAddressPort)

    pos_eye=''
    if max_index==0:
        pos_eye="RIGHT"
        color=[(0,0,0),(0,255,0)]
    elif max_index==1:
        pos_eye="CENTER"
        color=[(255,255,0),(255, 192, 203)]
    elif max_index==2:
        pos_eye = "LEFT"
        color = [(255,165,0), (0,0,0)]
    else:
        pos_eye = "CLOSED"
        color = [(255,165,0), (0,0,0)]

    return pos_eye,color

#set-up the mediapipe
with map_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:

    # for calculating fps
    start_time=time.time()

    while True: 
        frame_counter+=1
        ret,frame=camera.read()

        if not ret:
            break # if using break without the loop then gives error.

        # frame=cv2.flip(frame,1)
        # cv2-> use BGR channel
        # mediapipe -> use RGB channel
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb_frame) # stores the available landmarks

        # check whether the landmarks are available or not , if available then call landmarkDetection()
        if results.multi_face_landmarks:
            mesh_coords=landmarkDetection(frame,results,False) # True if want landmark on face

            # Blink detector counter
            right_coords=[mesh_coords[p] for p in RIGHT_EYE]
            left_coords=[mesh_coords[p] for p in LEFT_EYE]

            #crop eyes
            crop_right,crop_left=eyeExtractor(frame,right_coords,left_coords)

            eye_position_right,color=positionEstimator(crop_right)
            # utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 120), 2, color[0], color[1],8, 8)
            cv2.putText(frame,f'R: {eye_position_right}',(40, 120),FONTS,1,(0,255,255),2)

            eye_position_left, color = positionEstimator(crop_left)
            # utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 170), 2, color[0], color[1],8,8)
            cv2.putText(frame, f'l: {eye_position_left}', (40, 170), FONTS, 1, (0, 255, 255), 2)

            # starting voice indicator
            # if eye_position_left == "RIGHT" and eye_position_right!="LEFT" and pygame.mixer.get_busy() == 0:
            #     #starting counter
            #     counter_right+=1
            #     #resetting counters
            #     counter_left=0
            #     #playing voice
            #     voice_outside.play()
            #
            # if eye_position_left=="LEFT" and eye_position_right!="RIGHT"and pygame.mixer.get_busy()==0:
            #     #starting counter
            #     counter_left+=1
            #     #resetting counters
            #     counter_right=0
            #     voice_outside.play()

        #calculate fps
        end_time=time.time()-start_time
        fps=frame_counter/end_time

        # frame=cv2.putText(frame,f'Fps:{round(fps,1)}',(20,50),FONTS,1,(0,255,0),1)

        #for display frame
        cv2.imshow('Galanto_Innovation',frame)

        key=cv2.waitKey(1)
        if key==ord("m"):
            break

    camera.release()
    cv2.destroyAllWindows()
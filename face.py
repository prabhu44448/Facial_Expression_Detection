import cv2
import numpy as np
import dlib
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
import os



for tem in os.listdir("C:/Users/DELL/Desktop/database3/testing"):
    p=0
    for temp in os.listdir("C:/Users/DELL/Desktop/database3/testing/"+tem):
        p=p+1
        src="C:/Users/DELL/Desktop/database3/testing/"+tem+"/"+temp
        dst="C:/Users/DELL/Desktop/database3/testing/"+tem+"/"+tem+str(p)+".png"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        frame = cv2.imread(src)
        print(src)
        print(dst)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            pts = np.array([[x1,y1],[x2,y2]])
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = gray[y:y+h, x:x+w].copy()
            landmarks = predictor(gray, face)
            l=[]
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                l.append([x,y])
                a=cv2.circle(frame, (x, y), 2, (0,0,0), -1)   
            frame[:]=[255,255,255]    
            #print(l)
            for i in range(16):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)

            for i in range(17,26):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)
            cv2.line(frame, (l[42][0],l[42][1]), (l[47][0],l[47][1]), (0,0,0),2)
            cv2.line(frame, (l[36][0],l[36][1]), (l[41][0],l[41][1]), (0,0,0),2)
            #cv2.line(frame, (l[0][0],l[0][1]), (l[17][0],l[17][1]), (0,0,0),2)
            #cv2.line(frame, (l[26][0],l[26][1]), (l[16][0],l[16][1]), (0,0,0),2)
            for i in range(27,35):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)
            
            for i in range(48,67):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)
            for i in range(42,47):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)
            for i in range(36,41):
                cv2.line(frame, (l[i][0],l[i][1]), (l[i+1][0],l[i+1][1]), (0,0,0),2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            cv2.imwrite(dst,frame)

       
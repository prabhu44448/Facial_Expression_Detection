import cv2
import numpy as np
from PIL import Image,PngImagePlugin
import os
import io
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("cathy.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,3)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
    area=(x,y,w,h)
    print(area)
    crop_img = gray[y:y+h, x:x+w]
    cv2.imwrite("f.png",crop_img)
    cv2.imshow('imge',img)
    k = cv2.waitKey(1)
    if k == 27:
        break
                #img.release()

cv2.destroyAllWindows()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
import keyboard
import time


# In[2]:


def encase(img):
    img_copy = np.copy(img)
    faces_rects = haar_cascade_face.detectMultiScale(img_copy
                                                         , scaleFactor = 1.05, minNeighbors = 3);
    print(faces_rects)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img_copy


# In[6]:


haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0)

images = []
image = None


ret, frame = cap.read()
frame  = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) 

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(1,5):
    string = 'Press ' + str(i) + ' to capture image'
    
    empty = cv2.putText(np.ones(frame.shape), string , 
                (frame.shape[0] // 3, frame.shape[1] //3), 
                font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    images += [empty]
print(images[0].shape)
rect_imgs = list(images)

horizontal1 = np.hstack((images[0], images[1]))
horizontal2 = np.hstack((images[2], images[3]))
vertical = np.vstack((horizontal1, horizontal2))

cv2.imshow('Faces', vertical)
count = 0

while count != 4:
    keyPress = cv2.waitKey(0)
    if keyPress == ord('1'):
        count += 1
        ret, frame = cap.read()
        frame  = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) 
        images[0] = frame
        rect_imgs[0] = encase(frame)
        horizontal1 = np.hstack((rect_imgs[0], rect_imgs[1]))
    elif keyPress == ord('2'):
        count += 1
        ret, frame = cap.read()
        frame  = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) 
        images[1] = frame
        rect_imgs[1] = encase(frame)
        horizontal1 = np.hstack((rect_imgs[0], rect_imgs[1]))
    elif keyPress == ord('3'):
        count += 1
        ret, frame = cap.read()
        frame  = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) 
        images[2] = frame
        rect_imgs[2] = encase(frame)
        horizontal2 = np.hstack((rect_imgs[2], rect_imgs[3]))
    elif keyPress == ord('4'):
        count += 1
        ret, frame = cap.read()
        frame  = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)     
        images[3] = frame
        rect_imgs[3] = encase(frame)
        horizontal2 = np.hstack((rect_imgs[2], rect_imgs[3]))
    elif keyPress == 27:
        cv2.destroyAllWindows()
        raise Exception("Window Closed")
    
    vertical = np.vstack((horizontal1, horizontal2))
    cv2.imshow('Faces', vertical)
    cv2.waitKey(1)
    
print("Done")
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





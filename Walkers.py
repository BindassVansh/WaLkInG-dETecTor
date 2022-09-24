import cv2

body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
video = cv2.VideoCapture('E:\WhiteHatJr Projects\Face Recoginization\second\PRO-C106-ProjectSolution-main\walking.avi')

while True :
    ret,image = video.read()
    bandw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bodies = body.detectMultiScale(bandw,1.2,3)
    for (x,y,w,h) in bodies :
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('WAlking Detector',image)
    if cv2.waitKey(1) == 32 :
        break





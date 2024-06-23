import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# trained_face_data=cv2.CascadeClassifier("D:\Dev\Coding\dev.Personal\AI With Python\haarcascade_frontalface_default.xml")

webcam=cv2.VideoCapture(0)#"0" is added to get footage from default cam(primary camera) 1 for secondary camera


#Iterate over frames
while True:
    successful_frame_read,frame=webcam.read()

    greyscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    face_coordinates=trained_face_data.detectMultiScale(greyscaled_img)

    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)



    cv2.imshow("face detector with Python",frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
webcam.release()



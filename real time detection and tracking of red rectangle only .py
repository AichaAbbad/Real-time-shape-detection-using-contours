import cv2
import numpy as np

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

cv2.namedWindow("TRACK")
cv2.createTrackbar("L-H","TRACK",118,180,nothing)
cv2.createTrackbar("L-S","TRACK",97,255,nothing)
cv2.createTrackbar("L-V","TRACK",0,255,nothing)
cv2.createTrackbar("U-H","TRACK",180,180,nothing)
cv2.createTrackbar("U-S","TRACK",255,255,nothing)
cv2.createTrackbar("U-V","TRACK",255,255,nothing)

font = cv2.FONT_HERSHEY_SIMPLEX

while True :
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gray,(5,5),0)
    _ , thresh = cv2.threshold(blur,20 , 255,cv2.THRESH_BINARY)
    dialate = cv2.dilate(thresh, None, iterations = 3)

    hsv = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H","TRACK")
    l_s = cv2.getTrackbarPos("L-S", "TRACK")
    l_v = cv2.getTrackbarPos("L-V", "TRACK")
    u_h = cv2.getTrackbarPos("U-H", "TRACK")
    u_s = cv2.getTrackbarPos("U-S", "TRACK")
    u_v = cv2.getTrackbarPos("U-V", "TRACK")


    lower_red = np.array([l_h,l_s,l_v])
    upper_red = np.array([u_h,u_s,u_v])

    mask = cv2.inRange(hsv,lower_red, upper_red)
    kernal = np.ones((2,2), np.uint8)
    mask = cv2.erode(mask, kernal)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours :
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 400 :
            (x, y, w, h) = cv2.boundingRect(cnt)

            if cv2.contourArea(cnt) < 700:
                continue

            if len(approx) == 4:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(mask, "rectangle",(x,y), font,1,(255,255,255),2)


        print(len(approx))

    cv2.imshow("video", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break


cap.release ()
cv2.destroyAllWindows()

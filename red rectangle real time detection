import cv2
import numpy as np

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("TRACK")
cv2.createTrackbar("L-H","TRACK",118,180,nothing)
cv2.createTrackbar("L-S","TRACK",97,255,nothing)
cv2.createTrackbar("L-V","TRACK",0,255,nothing)
cv2.createTrackbar("U-H","TRACK",180,180,nothing)
cv2.createTrackbar("U-S","TRACK",255,255,nothing)
cv2.createTrackbar("U-V","TRACK",255,255,nothing)

font = cv2.FONT_HERSHEY_SIMPLEX

while True :
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

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
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

            if len(approx) == 4:
                cv2.putText(mask, "rectangle",(x,y), font,1,(255,255,255),2)


        print(len(approx))

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break


cap.release ()
cv2.destroyAllWindows()

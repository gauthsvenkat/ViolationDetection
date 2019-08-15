import cv2

cap = cv2.VideoCapture('traffic_lights.mp4')

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 1333, 800)

ret, frame = cap.read()

while(ret):
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    else:
        ret, frame = cap.read()

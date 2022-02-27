import cv2

cap = cv2.VideoCapture("rtsp://admin:wellocean2020@10.102.63.88")
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

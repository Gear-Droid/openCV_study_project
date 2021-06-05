import cv2
import dev

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# 1280x720
# 1920x1080

while(True):
    ret, frame = cap.read()

    if frame is not None:
        new_frame = frame
        print(frame.shape)
        print(frame.dtype)
        # new_frame = dev.img_processing(frame)
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 1000, 1000)
        cv2.imshow('Video', new_frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('waka2.jpg', frame)

cap.release()
cv2.destroyAllWindows()

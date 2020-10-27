import cv2
import pprint


"""cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Frame', gray)

    # cv2.imshow('Video', frame)

    if frame is not None:
        print(frame.shape)
        print(type(frame))
        print(frame.dtype)
        cv2.imshow('Frame', frame)

        # pprint.pprint(frame)
        # pprint.pprint(gray)

    # plt.imshow(frame)
    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""


img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Image Properties")
print("- Number of Pixels: " + str(img.size))
print("- Shape/Dimensions: " + str(img.shape))

blue, green, red = cv2.split(img)

"""
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]
"""

cv2.imshow('1', img)

        """
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        drawing = np.zeros((threshold.shape[0], threshold.shape[1], 3), dtype=np.uint8)

        # Draw polygonal contour + bonding rects + circles
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

        # Show in a window
        cv2.imshow('Contours', drawing)
        """
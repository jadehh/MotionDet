import cv2
def drawLine(img,line_y1,line_y2,xmin,ymin,xmax,ymax):
    img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0))
    img = cv2.line(img, (xmin, ymin + line_y1), (xmax, ymin + line_y1), (0, 0, 255), 2)
    # img = cv2.line(img, (xmin, ymin + line_y2), (xmax, ymin + line_y2), (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    capture = cv2.VideoCapture("/home/jade/Downloads/2019-09-17_18-53-01-060_1080_QSCPDX67SX/2019-09-17_18-50-47.mp4")
    while True:
        ret,frame = capture.read()
        img = drawLine(frame,10,30, 0, 460, 900, 720)
        cv2.imshow("result",img)
        cv2.waitKey(0)
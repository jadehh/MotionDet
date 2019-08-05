import cv2
def drawLine(img,line_y1,line_y2,xmin,ymin,xmax,ymax):
    img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0))
    img = cv2.line(img, (xmin, ymin + line_y1), (xmax, ymin + line_y1), (0, 0, 255), 2)
    img = cv2.line(img, (xmin, ymin + line_y2), (xmax, ymin + line_y2), (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    capture = cv2.VideoCapture("videos/2019-07-05_15-09-37.mp4")
    while True:
        ret,frame = capture.read()
        img = drawLine(frame,10,30, 0, 430, 838, 720)
        cv2.imshow("result",img)
        cv2.waitKey(0)
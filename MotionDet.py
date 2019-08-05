import cv2
import numpy as np
import time

result_index = []


class motion_det(object):
    def __init__(self, line_y1, line_y2, xmin, ymin, xmax, ymax):

        self.line_y1 = line_y1
        self.line_y2 = line_y2
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.delta_threshold = 5
        self.background1 = None
        self.background2 = None
        self.frame_cache = []
        self.frame_id_cache = []

        self.num_cache1 = [0] * 11
        self.num_cache2 = [0] * 11

        self.last_in_id = 0

    def process2(self, img, idx):
        col_mean = img[self.line_y2, :]
        col_mean = col_mean.astype(np.int32)
        # if idx == 281:
        #     print(col_mean)
        pixel_pad = np.pad(col_mean, (0, 1), mode='edge')
        pixel_delta = (pixel_pad[1:] - pixel_pad[:-1])

        if self.background2 is None:
            self.background2 = pixel_delta
            delta = np.zeros_like(pixel_delta, dtype=np.uint8)
        else:
            delta = pixel_delta - self.background2
            self.background2 = pixel_delta
        d_mask = np.abs(delta) > self.delta_threshold
        foreground = (d_mask - 0) * 255
        num_pixel = np.count_nonzero(d_mask)
        return foreground, num_pixel

    def process1(self, img, idx):

        col_mean = img[self.line_y1, :]
        col_mean = col_mean.astype(np.int32)
        # if idx == 281:
        #     print(col_mean)
        pixel_pad = np.pad(col_mean, (0, 1), mode='edge')
        pixel_delta = (pixel_pad[1:] - pixel_pad[:-1])
        if self.background1 is None:
            self.background1 = pixel_delta
            delta = np.zeros_like(pixel_delta, dtype=np.uint8)
        else:
            delta = pixel_delta - self.background1
            self.background1 = pixel_delta
        d_mask = np.abs(delta) > self.delta_threshold
        foreground = (d_mask - 0) * 255
        num_pixel = np.count_nonzero(d_mask)
        return foreground, num_pixel

    def detect(self, frame, idx):
        assert frame is not None

        result_frame = []
        result_id = []
        self.frame_cache.append(frame)
        self.frame_id_cache.append(idx)
        if len(self.frame_id_cache) > 11:
            self.frame_id_cache.pop(0)
            self.frame_cache.pop(0)

        # frame_block = frame[500:720, 600:1280, :].copy()
        frame_block = frame[self.ymin:self.ymax, self.xmin:self.xmax, :].copy()
        frame_block = cv2.cvtColor(frame_block, cv2.COLOR_BGR2GRAY)
        frame_block = cv2.GaussianBlur(frame_block, (5, 5), 0)

        foreground1, num_pixel1 = self.process1(frame_block, idx)
        foreground2, num_pixel2 = self.process2(frame_block, idx)
        # print("E/MtionDet3: 正在处理第---"+str(idx)+"---帧,num_pixel1 = "+str(num_pixel1)+",num_pixel2 = "+str(num_pixel2))

        # print("num_pixel = "+str(num_pixel))
        self.num_cache1.append(num_pixel1)
        self.num_cache1.pop(0)
        self.num_cache2.append(num_pixel2)
        self.num_cache2.pop(0)

        num_frame = len(self.frame_cache)
        if num_frame < 10:
            return result_frame, result_id

        if (self.num_cache1[2] == 0 and
            self.num_cache1[3] == 0 and
            self.num_cache1[4] == 0) and \
                (self.num_cache1[5] > 0 and
                 self.num_cache1[6] > 0 and
                 self.num_cache1[7] > 0 and
                 sum(self.num_cache1[5:8]) > 15) or \
                (self.num_cache2[2] == 0 and
                 self.num_cache2[3] == 0 and
                 self.num_cache2[4] == 0) and \
                (self.num_cache2[5] > 0 and
                 self.num_cache2[6] > 0 and
                 self.num_cache2[7] > 0 and
                 sum(self.num_cache2[5:8]) > 15):
            self.last_in_id = self.frame_id_cache[8]
            range_list = range(2, 9)
            if num_frame < 9:
                range_list = range(0, num_frame)
            for i in range_list:
                frame_temp = self.frame_cache[i]
                frame_id = self.frame_id_cache[i]
                cv2.line(frame_temp, (self.xmin, self.ymin + self.line_y1), (self.xmax, self.ymin + self.line_y1),
                         (0, 0, 255), 2)
                result_frame.append(frame_temp)
                result_id.append(frame_id)

        if (self.num_cache1[2] > 0 and
                self.num_cache1[3] > 0 and
                self.num_cache1[4] > 0 and
                sum( self.num_cache1[2:5]) > 15 and
                self.num_cache1[5] == 0 and
                self.num_cache1[6] == 0 and
                self.num_cache1[7] == 0) or \
                (self.num_cache2[2] > 0 and
                self.num_cache2[3] > 0 and
                self.num_cache2[4] > 0 and
                sum( self.num_cache2[2:5]) > 15 and
                self.num_cache2[5] == 0 and
                self.num_cache2[6] == 0 and
                self.num_cache2[7] == 0):
            # print('--------------------------------')
            # print(self.frame_id_cache[5])
            # print(self.num_cache)
            for i in range(2, 9):
                frame_temp = self.frame_cache[i]
                frame_id = self.frame_id_cache[i]
                if frame_id not in result_id:
                    cv2.line(frame_temp, (self.xmin, self.ymin + self.line_y1), (self.xmax, self.ymin + self.line_y1),
                             (0, 0, 255), 2)
                    result_frame.append(frame_temp)
                    result_id.append(frame_id)

        if (np.argmax(np.array(self.num_cache1[2:9])) == 3 and self.num_cache1[5] >= 30) or np.argmax(np.array(self.num_cache2[2:9])) == 3 and self.num_cache2[5] >= 30:
            range_list = range(1, 10)
            if self.frame_id_cache[5] - self.last_in_id <= 5:
                range_list = range(1, 6)
            if num_frame < 10:
                range_list = range(0, num_frame)
            for i in range_list:
                frame_temp = self.frame_cache[i]
                frame_id = self.frame_id_cache[i]
                if frame_id not in result_id:
                    cv2.line(frame_temp, (self.xmin, self.ymin + self.line_y1), (self.xmax, self.ymin + self.line_y1),
                             (0, 0, 255), 2)
                    result_frame.append(frame_temp)
                    result_id.append(frame_id)

        if len(result_id) > 0:
            for i in range(len(result_id)):
                image_temp = cv2.cvtColor(result_frame[i], cv2.COLOR_BGRA2BGR)
                cv2.imwrite('images_1_h264/%d.jpg' % result_id[i], image_temp)
                # cv2.imshow("result",result_frame[i])
                # cv2.waitKey(0)
                if result_id[i] not in result_index:
                    result_index.append(result_id[i])

        return result_frame, result_id
        # image_temp = cv2.cvtColor(result_frame[i],cv2.COLOR_BGRA2BGR)
        # cv2.imwrite('images_1_h264/%d.jpg' % result_id[i], image_temp)

    def reset(self, line_y, xmin, ymin, xmax, ymax):
        self.line_y = line_y
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.delta_threshold = 5
        self.background = None
        self.frame_cache = []
        self.frame_id_cache = []

        self.num_cache1 = [0] * 11
        self.num_cache2 = [0] * 11
        self.last_in_id = 0


if __name__ == '__main__':

    import json
    import base64

    # 65
    motion_detect = motion_det(10, 30, 0, 430, 838, 720)

    video_path0 = 'video_2019-07-16_16-41-25.avi'
    cap0 = cv2.VideoCapture(video_path0)
    ret0, frame0 = cap0.read()  # 3.2ms
    count0 = cap0.get(cv2.CAP_PROP_FRAME_COUNT)

    ii = 0
    while True:
        start_time = time.time()
        ret0, frame0 = cap0.read()  # 3.2ms
        if ret0 == False:
            break
        motion_detect.detect(frame0, ii)  # 0.3ms
        ii += 1
    #
    for index in result_index:
        print("E/testMotion: 从队列中取出第----" + str(index) + "------帧满足并且写入视频中,写入耗时45ms")

# coding:utf-8
'''
@goal: core class to process THz image
@date:2017-12-12
@author:varuy322

'''
import threading

import cv2
import numpy as np
import struct

import time

from writeImage import writeFrame

CHANNEL = 36
AD_POINTS = 390
SKIP_NUMBER = 13  # odd-even check,abandon some points data
MEAN_P = np.array(
    [0.94842, 1.11917, 0.91444, 1.15527, 0.95980, 0.96443, 1.04827, 0.93914, 0.99276, 0.95750, 0.94730, 0.92896,
     0.86907, 1.05891, 0.91514, 1.07121, 1.13077, 0.9329, 1.00230, 0.8865, 1.27430, 1.02718, 1.07843, 1.03650, 1.08183,
     0.95521, 0.84331, 1.02106, 1.00440, 0.85391, 1.06220, 0.92360, 1.32204, 1.00000, 1.19023,
     1.00000], dtype='double')  # np.ones((1, 36), dtype='double')
M = 401
N = 185
REALSIZE = (200, 400)
# distortion correction coefficient
DIST_CORR_COEFF = np.array([0.1, 0.3, 0.6])
BACKGROUND_CORRECT = True

SUSPICIOUS_THRESHOLD = 70


class ImageProcess():
    def __init__(self):
        self.isBackBuffFull = False
        self.isDataBuffFull = False
        self.imgArray_H = AD_POINTS  # sample points
        self.imgArray_W = CHANNEL  # channel numbers
        self.isSuspicous = False  # detect suspicious object
        self.recvBuff = []  # buffer for background correct
        self.dataBuff = []  # buffer for mean process
        self.isBackCorrect = True
        self.alarm = False

    def triFrameMean(self, frameList, weight=DIST_CORR_COEFF):

        tmp = np.zeros(frameList[0].shape)
        for i in range(3):
            tmp += frameList[i] * weight[i]
        return tmp

    def getDataFrameList(self, data):

        # print("dataBuff length:", len(self.dataBuff))
        if len(self.dataBuff) == 3:
            self.isDataBuffFull = True
            # self.dataBuff.append(data)
            # return False, []
        else:
            self.dataBuff.append(data)
            if len(self.dataBuff) == 3:
                self.isDataBuffFull = True
                # print("Databuff is Full!")
            else:
                self.isDataBuffFull = False
                # frameList = self.dataBuff
                # self.dataBuff.clear()
                # return True, frameList

    def getBackFrameList(self, frame, frameNums):
        '''
        :param frame: get one frame data
        :param frameNums: collect frame numbers to append list
        :return: a list ,contain frameNum frames
        '''
        # print("recvBuff length:>>", len(self.recvBuff))
        if len(self.recvBuff) == frameNums:
            self.isBackBuffFull = True
        else:
            self.recvBuff.append(frame)
            if len(self.recvBuff) == frameNums:
                self.isBackBuffFull = True
                # print("Full")
            else:
                self.isBackBuffFull = False

    def divMeanMat(self, inMatNp, meanMatNp, mean_pNp):

        return (inMatNp - meanMatNp) * mean_pNp

    def distortionCorrect(self, imgNp):
        '''
        由于扫描机制影响，单列剔除前13行，双列剔除后13列
        :param img: [390,36]输入图像数组,np
        :return: [390-13,36]返回畸变校准后的数组,np
        '''
        result = np.empty((AD_POINTS - SKIP_NUMBER, CHANNEL), dtype='int16')
        for idx in range(len(imgNp[0])):
            if isOdd(idx):
                result[:, idx] = imgNp[SKIP_NUMBER:, idx]
            else:
                result[:, idx] = imgNp[:-SKIP_NUMBER, idx]
        return result

    def img_Normalization(self, srcImgNp):
        maxValue = np.max(srcImgNp).astype('float')
        minValue = np.min(srcImgNp).astype('float')
        # print("srcImgNp shape is:", srcImgNp.shape)
        # print("maxValue=%f, minValue=%f" % (maxValue, minValue))
        normImgNp = (255.0 * (srcImgNp - minValue) / (maxValue - minValue)).astype('uint8')
        if (maxValue - minValue) > SUSPICIOUS_THRESHOLD:
            self.isSuspicous = True
        normImgNp_rev = 255 - normImgNp
        return normImgNp, normImgNp_rev

    def img_Filter(self, srcImgNp, filterType="Box", meanDriftFilter=False):
        # median filter
        mediaBlur_ksize = 5
        mediaBlurResult = cv2.medianBlur(srcImgNp, mediaBlur_ksize)
        boxFilter_ksize = (7, 7)
        boxFilterResult = cv2.boxFilter(mediaBlurResult, -1, boxFilter_ksize)
        if meanDriftFilter == True:
            pass

        return boxFilterResult

    def img_Resize(self, srcImg, realSize=REALSIZE):
        result = cv2.resize(srcImg, realSize, interpolation=cv2.INTER_LANCZOS4)  # Lanczos插值
        return result

    def img_Segmentation(self, srcImg):
        rects = []
        self.alarm = False
        # gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
        gray = srcImg
        threshold, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
        # print("threshold : ", threshold, " binary :", binary.shape)
        binary1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        srcImg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i in range(len(contours)):
            '''
            contours为轮廓数组
            '''
            # 矩形框
            x, y, w, h = cv2.boundingRect(contours[i])
            # print("x=%d ,y=%d ,w=%d ,h=%d" % (x, y, w, h))
            # print("srcImg shape:>>", srcImg.shape)

            if h * w > 10 and (h < gray.shape[0] / 4 or w < gray.shape[1] / 4):
                cv2.rectangle(srcImg, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 红色画出矩形框
                rects.append((x, y, w, h))
                # cv2.drawContours(srcImg, contours, -1, (255, 0, 0), 3)
                # print("image bgr shape:", srcImg.shape)
        if rects:
            self.alarm = True
        else:
            self.alarm = False
        return srcImg

    def pseudoColor(self, inputData):
        pass


def isOdd(frame_num):
    '''
    # 1    global function
    :param frame_num:
    :return: if odd return True，otherwise False
    '''
    if (frame_num & 1 == 1):
        return True
    else:
        return False


def getAdData(frame_num, srcBytes):
    '''
    1.global function
    transform frame bytes to int16 ,excluse encoder value
    :param srcBytes: one frame bytes
    :return: frame_num and AD matrix;np.array;int16,390*36
    '''
    # print("len frame bytes >>%d" % len(srcBytes))
    # print("srcBytes:", srcBytes)
    adData = struct.unpack('!%sh' % (len(srcBytes) // 2), srcBytes)
    # [390,36],get int16 ad data for image processing
    # print("len of adData >>%d" % len(adData))
    # print("type of adData >>", type(adData))
    reshapeArr = np.reshape(adData, (AD_POINTS, CHANNEL + 1))
    # print(adData)
    srcMat = reshapeArr[:, :CHANNEL]
    # print(type(srcMat), " ", (srcMat.shape))
    # srcMatReversed = np.zeros(shape=srcMat.shape)
    if isOdd(frame_num):
        srcMat = srcMat[::-1]  # np.flipup(srcMat)
    # print(max(srcMat[1, :]))
    AdData = (frame_num, srcMat)
    return AdData


def getBackMeanMat(meanMatList):
    '''
    get background mean matrix
    :param meanMatList: multi-frame ,item type:tuple
    :param frameNum:
    :return:after multiframe mean processing
    '''
    meanBackMat = np.zeros((AD_POINTS, CHANNEL), dtype='double')
    frameNumbers = len(meanMatList)
    # print("meanBackLength>>", frameNumbers)

    for item in meanMatList:
        # result = getAdData(item[0], item[1])
        meanBackMat += item[1]
    meanBackMat /= frameNumbers
    # print(meanBackMat)

    return meanBackMat


def process(addr, dataFrame, frame_num, imagePro, thzServer):
    print('process frame:%s' % frame_num)
    st = time.time()
    cost_time = 0
    oneFrame = getAdData(frame_num, dataFrame)  # (frame_num,AdData[390,36])
    # print("type of oneFrame >>", type(oneFrame))
    # print("len of oneFrame >>", (oneFrame[1].shape))
    # end_time = time.clock()
    # print("function of getAdData elapse :%f seconds." % (end_time - start_time))  # about:2ms
    if imagePro.isBackCorrect:
        if not imagePro.isBackBuffFull:
            imagePro.getBackFrameList(oneFrame[1], 10)
            # print("!!!!")
        else:
            # print("length of recvBuf:", len(imagePro.recvBuff))
            # imagePro.getBackFrameList(oneFrame[1], 10)
            # # print(oneFrame[1])
            print("backFrame process!!!\n", imagePro.isBackBuffFull)
            thzServer.meanBackMat = getBackMeanMat(imagePro.recvBuff)
            imagePro.recvBuff.clear()
            imagePro.isBackCorrect = False
    else:
        # print("oneFrame[%d] length:%s" % (frame_num, str(oneFrame[1].shape)))
        # print("meanBackMat length:%s" % str(meanBackMat.shape))  # (390,36)
        # print("mean_p length:", len(MEAN_P))
        # print("mean_p :", (MEAN_P))
        subtractBackMean = imagePro.divMeanMat(oneFrame[1], thzServer.meanBackMat, MEAN_P)
        afterCorrectData = imagePro.distortionCorrect(subtractBackMean)
        # print("afterCorrectData:", afterCorrectData.shape)
        imagePro.getDataFrameList(afterCorrectData)
        if imagePro.isDataBuffFull == True:
            avgFrameData = imagePro.triFrameMean(imagePro.dataBuff, weight=DIST_CORR_COEFF)
            imagePro.dataBuff.pop(0)
            imagePro.isDataBuffFull = False

            normalData = imagePro.img_Normalization(afterCorrectData)
            # print("normalData:", normalData[0].shape)
            result = imagePro.img_Resize(normalData[0])
            # print("resized image shape:", result.shape)
            afterFiltering = imagePro.img_Filter(result, filterType='Box')
            # print("afterFiltering shape: ", afterFiltering.shape, "afterFiltering type",
            #      (afterFiltering.dtype))
            segImg = imagePro.img_Segmentation(afterFiltering)
            # print("segImg shape:", segImg.shape)
            threading.Thread(target=writeFrame, args=(frame_num, segImg, imagePro.alarm)).start()
            # filename, cost_time = writeFrame(frame_num, segImg)
            # print('writeFrame frame_num:%s, file name:%s' % (frame_num, filename))
            # if imagePro.alarm:
            #
    thzServer.chunks[addr].pop(frame_num)
    packet_loss_frame_nums = list(thzServer.chunks[addr].keys())
    if len(packet_loss_frame_nums) > 5:
        print(packet_loss_frame_nums)
        for i in packet_loss_frame_nums:
            if i < (frame_num - 5):
                thzServer.chunks[addr].pop(i)
    return time.time() - st
    # return time.time() - st - cost_time

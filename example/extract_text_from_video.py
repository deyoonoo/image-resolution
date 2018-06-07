__author__ = 'shaofeng'

from PIL import Image
from sr_factory.sr_image_factory import SRImageFactory
import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import cv2
import numpy as np
import os
import logging
import sys
import time
import logging, logging.handlers
import numpy as np

def nothing(*arg):
        pass
def setup_custom_logger(name):
    formatter = \
        logging.Formatter(fmt='%(asctime)s %(levelname)-2s %(message)s'
                          , datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.handlers.RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5)
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
logger = setup_custom_logger('logapp')

# Initial HSV GUI slider values to load on program start.
icol = (0, 0, 0, 255, 255, 90)    # Green
#icol = (18, 0, 196, 36, 255, 255)  # Yellow
#icol = (89, 0, 0, 125, 255, 255)  # Blue
#icol = (0, 100, 80, 10, 255, 255)   # Red
cv2.namedWindow('colorTest')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

def letter_example():
    image = Image.open("/tmp/save.jpg")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("/tmp/tmpabcd.png", "png")

def letter_example(opencv_im):
    image = opencv_im
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("/tmp/tmpabcd.png", "png")


def babyface_example():
    image = Image.open("../test_data/babyface_4.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/babyface_4x.png", "png")

def monarch_example():
    image = Image.open("../test_data/monarch.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("../test_data/monarch_3x.png", "png")

def cv2_crop_img(cv2_img,x,y,width,height):
        crop_img = cv2_img[y:y+height, x:x+width]
        return crop_img
def conver_cv2imgto_pilimg(cv2_im):
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return pil_im

if __name__ == '__main__':

    #video_name = '../../testg.mp4'
    video_name = '../../linhnd.mp4'
    cap = cv2.VideoCapture(video_name)
    if(cap.isOpened()):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        logger.info('Open video: %s successfully',video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info('fps of video: %d - dimension:  %d x %d',fps,width,height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info('total frame number: %d - second: %d - minute: %d',length,(length/fps),(length/fps/60))
        res = True
        sec = 1200

        while(res):
            sec = (sec +20 )
            frame_number = int(sec * fps)
            cap.set(1,frame_number);
            res, frame = cap.read()
            if(res == True):
                top_percen = 0.008
                left_percen = 0.8
                w_percen = 0.05
                h_percen = 0.02
                pos_x = int(left_percen*width)+85
                pos_y = int(top_percen*height)
                crop_w = int(w_percen*width)
                crop_h = int(h_percen*height)
                crop_img = cv2_crop_img(frame,pos_x,pos_y,crop_w,crop_h)

                #cv2.imwrite('/tmp/save.jpg',crop_img)
                letter_example(conver_cv2imgto_pilimg(crop_img))
                lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
                lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
                lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
                highHue = cv2.getTrackbarPos('highHue', 'colorTest')
                highSat = cv2.getTrackbarPos('highSat', 'colorTest')
                highVal = cv2.getTrackbarPos('highVal', 'colorTest')

                frame2 = cv2.imread('/tmp/tmpabcd.png')
                frameBGR = cv2.GaussianBlur(frame2, (7, 7), 0)
                #frameBGR = cv2.medianBlur(frame, 7)
                #frameBGR = cv2.bilateralFilter(frame, 15 ,75, 75)
                """kernal = np.ones((15, 15), np.float32)/255
                frameBGR = cv2.filter2D(frame, -1, kernal)"""
                
                # Show blurred image.
                #cv2.imshow('blurred', frameBGR)
                
                # HSV (Hue, Saturation, Value).
                # Convert the frame to HSV colour model.
                frameHSV = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
                
                # HSV values to define a colour range we want to create a mask from.
                colorLow = np.array([lowHue,lowSat,lowVal])
                colorHigh = np.array([highHue,highSat,highVal])
                mask = cv2.inRange(frameHSV, colorLow, colorHigh)
                #cv2.imwrite('/tmp/mask.jpg',mask)
                # Show the first mask
                cv2.imshow('mask', mask)
                text = pytesseract.image_to_string(mask, lang='eng', boxes=False,config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789/')
                print('mask-plain: '+text)

                # Cleanup the mask with Morphological Transformation functions
                #kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
                #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
                #result = cv2.bitwise_and(frame2, frame2, mask = mask)
                #result[np.where((result == [0,0,0]).all(axis = 2))] = [255,255,255]

                text = pytesseract.image_to_string((frame2), lang='eng', boxes=False,config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789/')
                print('text: '+text)

                cv2.imshow('colorTest', frame2)
                #cv2.imshow('colorTest1', swap)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
    #cv2.imwrite('tmp.jpg',result)
    #babyface_example()
    #monarch_example()

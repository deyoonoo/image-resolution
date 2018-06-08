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
import datetime
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

def letter_example(opencv_im):
    image = opencv_im
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("/tmp/tmpabcd.png", "png")

def cv2_crop_img(cv2_img,x,y,width,height):
        crop_img = cv2_img[y:y+height, x:x+width]
        return crop_img
def conver_cv2imgto_pilimg(cv2_im):
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return pil_im

before_string = '0/0/0'

def check_str(mask_string,final_string):
    if(mask_string.count('/')==0 and final_string.count('/')==0):
        return -2

    try:
        if(mask_string == final_string):
            before_thang,before_thua,before_hotro = [int(s) for s in before_string.split('/')]
            current_thang,current_thua,current_hotro = [int(s) for s in final_string.split('/')]
            if(final_string == before_string):
                return 1
            else:
                chenhlech_thang = current_thang -before_thang
                chenhlech_thua = current_thua - before_thua
                chenhlech_hotro = current_hotro - before_hotro
                if(chenhlech_thang>-1 and chenhlech_thang<6 and chenhlech_thua>=0 and chenhlech_thua<=1 and chenhlech_hotro>=0 and chenhlech_hotro<6):
                    return 2
    except:
        logger.error('loi tai functions check_str')
        return -3
    return -1

step_sec = 20
top_percen = 0.004
left_percen = 0.8665
w_percen = 0.05
h_percen = 0.02
start_second = 60

list_time_event = []

def run_video(path_video):
    video_name = '../../linhnd.mp4'
    global before_string
    cap = cv2.VideoCapture(video_name)
    if(cap.isOpened()):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        logger.info('Open video: %s successfully',video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info('fps of video: %d - dimension:  %d x %d',fps,width,height)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info('total frame number: %d - time: %s',length,str(datetime.timedelta(seconds=length/fps)))
        res = True
        sec = start_second
        pos_x = int(left_percen*width)
        pos_y = int(top_percen*height)
        crop_w = int(w_percen*width)
        crop_h = int(h_percen*height)

        while(res):
            sec = (sec +step_sec )
            frame_number = int(sec * fps)
            cap.set(1,frame_number);
            res, frame = cap.read()
            if(res == True):
                start_time = time.time()
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

                frameHSV = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

                colorLow = np.array([lowHue,lowSat,lowVal])
                colorHigh = np.array([highHue,highSat,highVal])
                mask = cv2.inRange(frameHSV, colorLow, colorHigh)
                #cv2.imwrite('/tmp/mask.jpg',mask)
                # Show the first mask
                cv2.imshow('mask', mask)
                text1 = pytesseract.image_to_string(mask, lang='eng', boxes=False,config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789/').replace(' ','')
                logger.debug('before string: %s',before_string)
                logger.info('text1: %s',text1)
                text2 = pytesseract.image_to_string((frame2), lang='eng', boxes=False,config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789/').replace(' ','')
                #print('text: '+text2)
                logger.info('text2: %s',text2)
                check_res = check_str(text1,text2)
                if(text2 == '0/0/0'):
                    before_string=text2
                if(check_res<0):
                    logger.warning('current time - %s img to text is not correct',str(datetime.timedelta(seconds=sec)))
                    if(check_res == -1):
                        sec = sec - step_sec +1


                else:
                    logger.info('current time - %s check string status: %s',str(datetime.timedelta(seconds=sec)),str(check_str(text1,text2)))
                    before_string = text2;
                    if(check_res == 2):
                        list_time_event.append(sec)
                cv2.imshow('colorTest', frame2)
                logger.info('%2.3f second',(time.time() - start_time))
                #cv2.imshow('colorTest1', swap)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
        logger.info('the end: %s',video_name)
    else:
        logger.info('Open video: %s error',video_name)



if __name__ == '__main__':

    #video_name = '../../testg.mp4'
    video_name = '../../linhnd.mp4'
    list_time_event =[]
    before_string = '0/0/0'
    run_video(video_name)
    for sec in list_time_event:
        logger.info(str(datetime.timedelta(seconds=sec)))

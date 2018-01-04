#coding=utf-8
import cv2
import numpy as np
import usb.core
import usb.backend.libusb1
import requests
import time
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)
backend = usb.backend.libusb1.get_backend(find_library=lambda x: "/usr/lib/libusb-1.0.so")
 # 
dev = usb.core.find(idVendor=0x18e3, idProduct=0x5031, backend=backend)
dev.ctrl_transfer(0x21,0x01,0x0800,0x0600,[0x50,0xff])
dev.ctrl_transfer(0x21,0x01,0x0f00,0x0600,[0x00,0xf6])
dev.ctrl_transfer(0x21,0x01,0x0800,0x0600,[0x25,0x00])
dev.ctrl_transfer(0x21,0x01,0x0800,0x0600,[0x5f,0xfe])
dev.ctrl_transfer(0x21,0x01,0x0f00,0x0600,[0x00,0x03])
dev.ctrl_transfer(0x21,0x01,0x0f00,0x0600,[0x00,0x02])
dev.ctrl_transfer(0x21,0x01,0x0f00,0x0600,[0x00,0x12])
dev.ctrl_transfer(0x21,0x01,0x0f00,0x0600,[0x00,0x04])
dev.ctrl_transfer(0x21,0x01,0x0800,0x0600,[0x76,0xc3])
dev.ctrl_transfer(0x21,0x01,0x0a00,0x0600,[4,0x00])   

firstFrame = None
window_size = 2
min_disp = 0
num_disp = 112
while(1):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 480), interpolation=cv2.CV_8SC1)
        #cv2.imshow("old",frame)
        #cv2.cvtColor(frame, frame, cv2.COLOR_BGR2GRAY);
        frame_left_old = frame[0:480,0:640]
        frame_left_old = cv2.GaussianBlur(frame_left_old, (21, 21), 0) 
        cv2.imshow("old_left",frame_left_old)
        frame_left = cv2.cvtColor(frame_left_old,  cv2.COLOR_BGR2GRAY);
        frame_right_old = frame[0:480,640:1280]
        frame_right_old = cv2.GaussianBlur(frame_right_old, (21, 21), 0)
        frame_right = cv2.cvtColor(frame_right_old,  cv2.COLOR_BGR2GRAY);
        #stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        stereo = cv2.StereoSGBM_create(minDisparity = 0,
            numDisparities = num_disp,
            blockSize = 15,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 24,
            uniquenessRatio = 11,
            speckleWindowSize = 175,
            speckleRange = 46
        )
        disp = stereo.compute(frame_left, frame_right).astype(np.float32) / 16.0
        #h, w = frame_left.shape[:2]
        #f = 0.8*w
        #mask = disp > disp.min()
        #Q = np.float32([[1, 0, 0, -0.5*w],
        #        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
        #        [0, 0, 0,     -f], # so that y-axis looks up
        #        [0, 0, 1,      0]])
        #points = cv2.reprojectImageTo3D(disp, Q)
        #out_points = points[mask]
        #print(out_points)
        #colors = cv2.cvtColor(frame_left_old, cv2.COLOR_BGR2RGB)
        #print(colors)
        #write_ply('out.ply',out_points,colors)
        #disparity = stereo.compute(frame_left,frame_right)
        #disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #print('-----------------')
        #print(disp[1,120:])
        cv2.imshow("h",disp/num_disp)

        disp = np.where(disp > 60,disp,0)
        l1_new,l2_new = np.split(disp[:,120:],2,axis = 1)
        cv2.imshow('l1_new',l1_new)
        cv2.imshow('l2_new',l2_new)        
        #l1,l2 = np.vsplit(disp[:,120:], 2)
        #l1_new = np.where(l1 > 90, l1, 0) 
        #l2_new = np.where(l2 > 90, l2, 0)
        l1_sum = np.sum(l1_new)
        l2_sum = np.sum(l2_new)
        if abs(l1_sum - l2_sum) < 1000000:
            #continue
            print('gogogo')
            continue
        elif l1_sum > l2_sum :
            print("右转",l1_sum - l2_sum)
        else:
            print("左转",l1_sum - l2_sum)
        #print("l1:",np.sum(l1_new)/1000)
        #print("l2:",np.sum(l2_new)/1000)
        
        #print("left:")
        #print( np.mean(l1))
        #print("right:")
        #print( np.mean(l2))
        #cv2.imshow("h",disp/num_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()


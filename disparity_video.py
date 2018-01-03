import numpy as np
import cv2
import usb.core
import usb.backend.libusb1

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

#l_camera = cv2.VideoCapture(0)
#r_camera = cv2.VideoCapture(1)


# create windows
cv2.namedWindow('left_Webcam', cv2.WINDOW_NORMAL)
cv2.namedWindow('right_Webcam', cv2.WINDOW_NORMAL)
cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)

blockSize = 40
def update(val = 0):
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print('computing disparity...')
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    cv2.imshow('left', gray_left)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    

while(cv2.waitKey(1) & 0xFF != ord('q')):
    #ret1, left_frame = l_camera.read()
    #ret2, right_frame = r_camera.read()
    window_size = 5
    min_disp = 16
    num_disp = 192-min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 480), interpolation=cv2.CV_8SC1)   
    left_frame = frame[0:480,0:640] 
    right_frame = frame[0:480,640:1280]
    # our operations on the frame come here
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('left_Webcam', gray_left)
    cv2.imshow('right_Webcam', gray_right)
    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)    
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = P1,
        P2 = P2
    )
    '''
    stereo = cv2.StereoSGBM_create(minDisparity=1,
        numDisparities=16,
        blockSize=15,
        #uniquenessRatio = 10,
        speckleWindowSize = 10,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*blockSize**2,
        P2 = 32*3*blockSize**2)
        '''
    update()
    #disparity = stereo.compute(gray_left, gray_right)
    #disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imshow('disparity', disparity)
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

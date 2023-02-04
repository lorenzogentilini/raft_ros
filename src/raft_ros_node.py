#!/usr/bin/env python3
import sys
sys.path.append('/home/nvidia/catkin_ws/src/raft_ros/RAFT/core')

import rospy as rp
import numpy as np
import math as mt
import torch
import cv2

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder, forward_interpolate
from recordtype import recordtype
from sensor_msgs.msg import Image

RAFTArgs = recordtype('RAFTArgs', 'alternate_corr mixed_precision model small corr_levels corr_radius dropout')

class RaftWrapper:
    def __init__(self):
        # Attributes
        self.prevImg = Image()
        self.actualImg = Image()
        self.flow_prev = None
        self.count = 0

        self.CVDepthToNumpy = {cv2.CV_8U: 'uint8', cv2.CV_8S: 'int8', cv2.CV_16U: 'uint16',
                               cv2.CV_16S: 'int16', cv2.CV_32S:'int32', cv2.CV_32F:'float32',
                               cv2.CV_64F: 'float64'}
        self.NumpyToCVType  = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                               'int16': '16S', 'int32': '32S', 'float32': '32F',
                               'float64': '64F'}
        # Subscribers
        self.imageSub = rp.Subscriber("/d400/color/image_raw", Image, self.imageCallback, queue_size=1)

        # Publishers
        self.optFlowPub = rp.Publisher('/opt_flow', Image, queue_size=1)

        # Load RAFT Model
        args = RAFTArgs(alternate_corr=False, mixed_precision=True, model='/home/nvidia/catkin_ws/src/raft_ros/RAFT/models/raft-things.pth', small=False, corr_levels=0, corr_radius=0, dropout=0)
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))
        self.model.cuda()
        self.model.eval()

        print("Model Loaded!")

        # Img Proc
        detector_params = cv2.SimpleBlobDetector_Params()

        detector_params.minThreshold = 50
        detector_params.maxThreshold = 200
        detector_params.minDistBetweenBlobs = 100

        # Filter by Color
        detector_params.filterByColor = True
        detector_params.blobColor = 0
        # Filter by Inertia
        detector_params.filterByInertia = False
        detector_params.maxInertiaRatio = 1.0
        detector_params.minInertiaRatio = 0.8
        # Filter by Area
        detector_params.filterByArea = True
        detector_params.minArea = 10
        detector_params.minArea = 200
        # Filter by Convexity
        detector_params.filterByConvexity = False
        detector_params.minConvexity = 0.8
        detector_params.maxConvexity = 1
        # Filter by Circularity
        detector_params.filterByCircularity = False
        detector_params.maxCircularity = 1.0
        detector_params.minCircularity = 0.8

        self.detector = cv2.SimpleBlobDetector_create(detector_params)

    def imageCallback(self, msg):
        # Memorize Upcoming Images
        if self.count < 2:
            self.prevImg = self.actualImg
            self.actualImg = msg
            self.count += 1
    
    def execute(self):
        if self.count < 2:
            return

        with torch.no_grad():
            img1 = self.msgToCVImg(self.prevImg)
            img2 = self.msgToCVImg(self.actualImg)

            # Convert Input Images
            img1 = self.getTorchImage(img1)
            img2 = self.getTorchImage(img2)

            # Resizing
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            
            # Run Model
            flow_low, flow_up = self.model(img1, img2, iters=10, flow_init=self.flow_prev, test_mode=True)

            # Map Flow to RGB Image
            flow_up = padder.unpad(flow_up[0]).permute(1,2,0).cpu().numpy()
            self.flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            flow_up = flow_viz.flow_to_image(flow_up)

            # Flow Post-Elaboration
            gray_flow = cv2.cvtColor(flow_up, cv2.COLOR_BGR2GRAY)

            #blobs = self.detector.detect(gray_flow)
            flow_up = cv2.drawKeypoints(flow_up, blobs, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Publish Results
            self.optFlowPub.publish(self.CVImgToMsg(flow_up, "rgb8"))
            self.count = 0

    def getTorchImage(self, surce_img):
        img = torch.from_numpy(np.array(surce_img)).permute(2, 0, 1).float()
        return img[None].cuda()

    #### MSG/CV Conversion ################################
    def CVImgToMsg(self, img, enc = "passthrough"):
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]

        if len(img.shape) < 3:
            cv_type = self.fromDTypeToCVType(img.dtype, 1)
        else:
            cv_type = self.fromDTypeToCVType(img.dtype, img.shape[2])

        if enc == "passthrough":
            msg.encoding = cv_type
        else:
            msg.encoding = enc

        if img.dtype.byteorder == '>':
            msg.is_bigendian = True

        msg.data = img.tostring()
        msg.step = len(msg.data) // msg.height

        return msg

    def msgToCVImg(self, msg):
        dtype, n_channels = self.fromEcodingToDType(msg.encoding)
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')

        if n_channels == 1:
            im = np.ndarray(shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data)
        else:
            if(type(msg.data) == str):
                im = np.ndarray(shape=(msg.height, msg.width, n_channels), dtype=dtype, buffer=msg.data.encode())
            else:
                im = np.ndarray(shape=(msg.height, msg.width, n_channels), dtype=dtype, buffer=msg.data)

        # Chech Byteorder
        if msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()
        
        return im

    def fromDTypeToCVType(self, dtype, channels):
        return '%sC%d' % (self.NumpyToCVType[dtype.name], channels)

    def fromEcodingToDType(self, enc):
        return self.fromCVTypeToDType(self.fromEncodingToCVType(enc))

    def fromCVTypeToDType(self, cvtype):
        return self.CVDepthToNumpy[self.CvMatDepth(cvtype)], self.CvMatChannels(cvtype)

    def CvMatDepth(self, flag):
        return flag & ((1 << 3) - 1)

    def CvMatChannels(self, flag):
        return ((flag & (511 << 3)) >> 3) + 1

    def fromEncodingToCVType(self, enc):
        if(enc == "bgr8"):
            return cv2.CV_8UC3
        if(enc == "mono8"):
            return cv2.CV_8UC1
        if(enc == "rgb8"):
            return cv2.CV_8UC3
        if(enc == "mono16"):
            return cv2.CV_16UC1
        if(enc == "bgr16"):
            return cv2.CV_16UC3
        if(enc == "rgb16"):
            return cv2.CV_16UC3
        if(enc == "bgra8"):
            return cv2.CV_8UC4
        if(enc == "rgba8"):
            return cv2.CV_8UC4
        if(enc == "bgra16"):
            return cv2.CV_16UC4
        if(enc == "rgba16"):
            return cv2.CV_16UC4
        if(enc == "bayer_rggb8"):
            return cv2.CV_8UC1
        if(enc == "bayer_bggr8"):
            return cv2.CV_8UC1
        if(enc == "bayer_gbrg8"):
            return cv2.CV_8UC1
        if(enc == "bayer_grbg8"):
            return cv2.CV_8UC1
        if(enc == "bayer_rggb16"):
            return cv2.CV_16UC1
        if(enc == "bayer_bggr16"):
            return cv2.CV_16UC1
        if(enc == "bayer_gbrg16"):
            return cv2.CV_16UC1
        if(enc == "bayer_grbg16"):
            return cv2.CV_16UC1
        if(enc == "yuv422"):
            return cv2.CV_8UC2
    #### END ################################
    
# Main Function
if __name__ == '__main__':
    rp.init_node('raft_ros_node', anonymous = False)
    
    # Initialize Object
    RaftRos = RaftWrapper()

    # Spin
    while not rp.is_shutdown():
        RaftRos.execute()

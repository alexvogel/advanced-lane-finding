#!/usr/bin/python

"""
    Project 4
    Udacity Self-Driving-Car-Engineer Nanodegree

    Advanced Lane Line Detection
    
    1) Camera Calibration (Create Camera Matrix)
    2) Distortion Correction of Images
    * Binary Image By Thresholding Color Transformation, Gradients, etc.
    * Perspective Transform to "birds-eye view"
    * Detect Lane Pixels And Find The Lane Boundary
    * Determine The Curvature Of The Lane And Vehicle Position With Respect To Center
    * Warp Detected Lane Boundaries Back Onto Original Image
    * Output Visual Display Of
        ** Lane Boundaries
        ** Numerical Estimation
        ** Lane Curvature
        ** Vehicle Position
"""    

import os
import sys
import argparse
from time import time

# add lib to path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
from helper_lane_lines import *

# setting etc dir
etcDir = os.path.dirname(os.path.realpath(__file__))+'/../etc'

version = "0.1"
date = "2017-04-27"

# Definieren der Kommandozeilenparameter
parser = argparse.ArgumentParser(description='a tool for detecting lane lines in images and videos',
                                 epilog='author: alexander.vogel@prozesskraft.de | version: ' + version + ' | date: ' + date)
parser.add_argument('--image', metavar='PATH', type=str, nargs='*', required=False,
                   help='image from a front facing camera. to detect lane lines')
parser.add_argument('--video', metavar='PATH', type=str, nargs='*', required=False,
                   help='video from a front facing camera. to detect lane lines')
parser.add_argument('--visLog', action='store_true', default=False,
                   help='for debugging or documentation of the pipeline. use only with one image. creates an output image for every step of the pipeline.')
parser.add_argument('--outDir', metavar='PATH', action='store', default='output_directory_'+str(time()),
                   help='directory for output data. must not exist at call time.')
parser.add_argument('--calDir', metavar='PATH', action='store', required=False, default=etcDir + '/camera_cal',
                   help='directory for camera calibration images. directory must only contain chessboard 9x6 calibration images.')

args = parser.parse_args()

errors = 0

# check whether image or video was supplied
if not args.image and not args.video:
    log('error', 'you need to provide at least one image or video. try --help for help.')
    errors += 1

# check if all provided images exist
if args.image:
    for path in args.image:
        if not os.path.isfile(path):
            log('error', 'image does not exist:'+ path)
            errors += 1

# check if all provided videos exist
if args.video:
    for path in args.video:
        if not os.path.isfile(path):
            log('error', 'video does not exist:'+ path)
            errors += 1
        
# check if calDir does NOT exist
if not os.path.isdir(args.calDir):
    log('error', 'directory with camera calibration images does not exist: ' + args.calDir)
    errors += 1

# check if outDir does exist
if os.path.isdir(args.outDir):
    log('error', 'output directory already exists. please delete or rename:' + args.outDir)
    errors += 1

if errors > 0:
    log('fatal', str(errors) + ' error(s) occured. please correct and try again.')
    sys.exit(1)
    
log('info', '--outDir='+args.outDir)
log('info', '--calDir='+args.calDir)

#======================
#
# 1) Create Output Directory
#
#----------------------


#======================
#
# 2) Camera Calibration
#
#----------------------

ret, mtx, dist, rvecs, tvecs = calibrateCamera(args.calDir)

#======================
#
# 2) Distortion Correction of Images
#
#----------------------






for imagePath in args.image:
    
    # read image
    img = mpimg.imread(imagePath)
    
    plt.imshow(img)
    plt.show()

    result = laneLinePipeline(img, mtx, dist, args.outDir, args.visLog, sobel_kernel=5, mag_sobelxy_thresh=(30, 100), hls_thresh=(170, 255))

    
    
    


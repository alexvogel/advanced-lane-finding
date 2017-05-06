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
from moviepy.editor import VideoFileClip

# add lib to path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
from helper_lane_lines import *
from line import Line

# setting etc dir
etcDir = os.path.dirname(os.path.realpath(__file__))+'/../etc'

version = "0.1"
date = "2017-04-27"

# Definieren der Kommandozeilenparameter
parser = argparse.ArgumentParser(description='a tool for detecting lane lines in images and videos',
                                 epilog='author: alexander.vogel@prozesskraft.de | version: ' + version + ' | date: ' + date)
parser.add_argument('--image', metavar='PATH', type=str, nargs='?', required=False,
                   help='image from a front facing camera. to detect lane lines')
parser.add_argument('--video', metavar='PATH', type=str, nargs='?', required=False,
                   help='video from a front facing camera. to detect lane lines')
parser.add_argument('--visLog', metavar='INT', type=int, action='store', default=False,
                   help='for debugging or documentation of the pipeline. \
                   1=undistorted image \
                   2=grayscale \
                   3=binary mask magnitude sobel xy \
                   4=hls binary mask \
                   5=combination of binary masks \
                   6=unwarped binary with polygon \
                   7=warped binary with polygon \
                   8=warped binary \
                   9=histogram \
                   10=detected lines \
                   11=undistorted with detected lines \
                   12=result with text' \
                   )
parser.add_argument('--format', metavar='STRING', type=str, action='store', default='normal',
                   help='setting for result image. --format=collage4, --format=collage9')
parser.add_argument('--outDir', metavar='PATH', action='store', default='output_directory_'+str(time()),
                   help='directory for output data. must not exist at call time.')
parser.add_argument('--calDir', metavar='PATH', action='store', required=False, default=etcDir + '/camera_cal',
                   help='directory for camera calibration images. directory must only contain chessboard 9x6 calibration images.')

args = parser.parse_args()

map_int_name = {    
                    0: '00_original',
                    1: '01_undist',
                    2: '02_gray',
                    3: '03_binary_sobelxy',
                    4: '04_binary_hls',
                    5: '05_combined_binaries',
                    6: '06_transform1',
                    7: '07_transform2',
                    8: '08_warped_binary',
                    9: '09_histogram',
                    10:'10_detect_lines',
                    11:'13_undist_with_polyfit',
                    12:'14_result_with_text',
                    False: '99_result',
                }



errors = 0

# check whether image or video was supplied
if not args.image and not args.video:
    log('error', 'you need to provide at least one image or video. try --help for help.')
    errors += 1

# check if all provided images exist
if args.image:
    if not os.path.isfile(args.image):
        log('error', 'image does not exist:'+ args.image)
        errors += 1

# check if all provided videos exist
if args.video:
    if not os.path.isfile(args.video):
        log('error', 'video does not exist:'+ args.video)
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

log('info', '--visLog=' + str(args.visLog))
log('info', '--format=' + args.format)




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

leftLine = Line()
rightLine = Line()

def process_image(img, leftLine=leftLine, rightLine=rightLine):
    result, leftLine, rightLine = laneLinePipeline(img, mtx, dist, args.outDir, args.visLog, leftLine, rightLine, args.format, sobel_kernel=5, mag_sobelxy_thresh=(30, 100), hls_thresh=(170, 255))
    return result



if args.image:
    
    # read image
    img = mpimg.imread(args.image)
    result, leftLine, rightLine = laneLinePipeline(img, mtx, dist, args.outDir, args.visLog, leftLine, rightLine, args.format, sobel_kernel=5, mag_sobelxy_thresh=(30, 100), hls_thresh=(170, 255))
    
    print(map_int_name[args.visLog])
    writeImage(result, args.outDir, map_int_name[args.visLog], cmap=None)

if args.video:
    video_output = args.outDir + '/video_out.mp4'
    clip1 = VideoFileClip(args.video, audio=False)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    
    if not os.path.isdir(args.outDir):
        log('info', 'creating output directory: ' + args.outDir)
        os.mkdir(args.outDir)

    white_clip.write_videofile(video_output, audio=False)
    


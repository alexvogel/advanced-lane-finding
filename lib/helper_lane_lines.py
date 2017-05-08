import os
import datetime
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# helper functions
def log(stage, msg):
    '''
        function to print out logging statement in this format:
        
        format
        <time> : <stage> : <msg>
        
        example:
        2017-04-28 12:48:45 : info : chess board corners found in image calibration20.jpg
    '''
    
    
    print(str(datetime.datetime.now()).split('.')[0] + " : " + stage + " : " + msg)

def writeImage(item, dir, basename, cmap=None):
    '''
        write an image(s) to file
        fig: matplotlib figure or image as an numpy array
        filePath: path to write the image to
    '''
    
    
    # create dir if nonexistent
    if not os.path.isdir(dir):
        log('info', 'creating output directory: ' + dir)
        os.mkdir(dir)
    
    # if numpy array - write it
    #if type == 

    # define filename
    file = dir + '/' + basename + '.png'
    log('info', 'writing image: ' + file)

    # if ndarray
    if isinstance(item, np.ndarray):
        if len(item.shape) == 1:
            fig = plt.figure(1)
            ax = plt.axes()
            plt.plot(item)
            fig.savefig(file)
        else:
            mpimg.imsave(file, item, cmap=cmap)
    else:
        fig = item
        fig.savefig(file)
    
    plt.clf()

def laneLinePipeline(rgb, mtx, dist, outDir, retNr, leftLine, rightLine, format, sobel_kernel=5, mag_sobelxy_thresh=(30, 100), hls_thresh=(170, 255)):
    '''
        processes an image from input to output in finding a lane line
        img: input image in bgr colorspace
        mtx: camera calibration matrix
        dist: camera distortion coefficients
        outDir: output directory path
        retNr: return the result of a certain step of the pipeline
        leftLine: tracking instance for the left line
        rightLine: tracking instance for the right line
        sobel_kernel: size of the sobel kernel
        mag_sobelxy_thresh: tuple of min and max threshold for the binary generation
        return: image with detected-lanes-overlay
    '''

    # store for the intermediate steps of the processing pipeline
    imageBank = {}
    imageBank[0] = rgb

    if retNr is 0:
        return rgb

    # undistort
    #undistort = np.copy(rgb)
#    log('debug', 'undistort image')
    rgb_undistort = cv2.undistort(rgb, mtx, dist, None, mtx)
    imageBank[1] = rgb_undistort
    if retNr is 1:
        return rgb_undistort, leftLine, rightLine

    # convert to grayscale
#    log('debug', 'convert to grayscale')
    gray = cv2.cvtColor(rgb_undistort, cv2.COLOR_RGB2GRAY)
    gray_as_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    imageBank[2] = gray_as_rgb
    if retNr is 2:
        return gray_as_rgb, leftLine, rightLine

    # create binary mask of the thresholded magnitude of sobel operator
#    log('debug', 'create binary mask with abs_sobelxy operator')
    binary_output_abs_sobelxy = getBinaryMagSobelXY(gray, sobel_kernel, mag_sobelxy_thresh)
    binary_output_abs_sobelxy_as_rgb = cv2.cvtColor(binary_output_abs_sobelxy * 255, cv2.COLOR_GRAY2RGB)
    imageBank[3] = binary_output_abs_sobelxy_as_rgb
    if retNr is 3:
        return binary_output_abs_sobelxy_as_rgb, leftLine, rightLine

    # create binary mask of the thresholded s of the colorspace hls
#    log('debug', 'create binary mask with the s of the HLS color space')
    binary_output_s_of_hls = getBinarySHls(rgb_undistort, hls_thresh)
    binary_output_s_of_hls_as_rgb = cv2.cvtColor(binary_output_s_of_hls * 255, cv2.COLOR_GRAY2RGB)
    imageBank[4] = binary_output_s_of_hls_as_rgb
    if retNr is 4:
        return binary_output_s_of_hls_as_rgb, leftLine, rightLine

    # create a combined binary (gradient and colorspace)
#    log('debug', 'combine sobel and s binary mask to a single binary mask')
    binary_combined = combineBinaries([binary_output_abs_sobelxy, binary_output_s_of_hls])
    binary_combined_as_rgb = cv2.cvtColor(binary_combined * 255, cv2.COLOR_GRAY2RGB)
    imageBank[5] = binary_combined_as_rgb
    if retNr is 5:
        return binary_combined_as_rgb, leftLine, rightLine

    # warp image from camera view to birds eye view
#    log('debug', 'transform image from camera view to birds eye view')
    binary_combined_warped, figs, M, Minv = transformToBirdsView(binary_combined)

    figs[0].canvas.draw() # draw the canvas, cache the renderer
    tmp = np.fromstring(figs[0].canvas.tostring_rgb(), dtype=np.uint8, sep='')
    unwarped_binary_with_polygon = cv2.resize(tmp.reshape(figs[0].canvas.get_width_height()[::-1] + (3,)), (rgb_undistort.shape[1], rgb_undistort.shape[0]))
    imageBank[6] = unwarped_binary_with_polygon
    if retNr is 6:
        return unwarped_binary_with_polygon, leftLine, rightLine

    figs[1].canvas.draw() # draw the canvas, cache the renderer
    tmp = np.fromstring(figs[1].canvas.tostring_rgb(), dtype=np.uint8, sep='')
    warped_binary_with_polygon = cv2.resize(tmp.reshape(figs[1].canvas.get_width_height()[::-1] + (3,)), (rgb_undistort.shape[1], rgb_undistort.shape[0]))
    imageBank[7] = warped_binary_with_polygon
    if retNr is 7:
        return warped_binary_with_polygon, leftLine, rightLine
     
    binary_combined_warped_as_rgb = cv2.cvtColor(binary_combined_warped * 255, cv2.COLOR_GRAY2RGB)
    imageBank[8] = binary_combined_warped_as_rgb
    if retNr is 8:
        return binary_combined_warped_as_rgb, leftLine, rightLine
    
    # take a histogram along all the columns in the lower half of the image
    histogram = np.sum(binary_combined_warped[binary_combined_warped.shape[0]//2:, :], axis=0)
    plt.clf()
    fig = plt.figure(1)
    ax = plt.axes()
    plt.plot(histogram)
    fig.canvas.draw()
    tmp = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    histogram_as_rgb = cv2.resize(tmp.reshape(fig.canvas.get_width_height()[::-1] + (3,)), (rgb_undistort.shape[1], rgb_undistort.shape[0]))
    imageBank[9] = histogram_as_rgb
    if retNr is 9:
        return histogram_as_rgb, leftLine, rightLine

    # calc lane width
    widthMeter = calcLaneWidth(leftLine, rightLine)

    # detect Lines
    if leftLine.isDetected() and rightLine.isDetected() and isLaneWidthPlausible(widthMeter):
#    if False:
         detected_or_sliding_window, left_poly_coeff, right_poly_coeff = findLinesSimple(binary_combined_warped, leftLine.getBestPolyCoeff(), rightLine.getBestPolyCoeff())
    else:
        # finding the lines with sliding window
        detected_or_sliding_window, left_poly_coeff, right_poly_coeff = findLines(binary_combined_warped)

    imageBank[10] = detected_or_sliding_window
    if retNr is 10:
        return detected_or_sliding_window, leftLine, rightLine

    # if one line is faulty recognized and jumps 
    left_poly_coeff_smooth, right_poly_coeff_smooth, correctionStatement = smoothPolyCoeff(leftLine, rightLine, left_poly_coeff, right_poly_coeff)

    # set the coeffs
    leftLine.setDetected(True)
    rightLine.setDetected(True)
    leftLine.setCurrentPolyCoeff(left_poly_coeff_smooth)
    rightLine.setCurrentPolyCoeff(right_poly_coeff_smooth)
    

    # generate x-y-values for plotting the lines
    left_line_x, right_line_x, both_lines_y = generateLineXYValues(rgb, leftLine.getBestPolyCoeff(), rightLine.getBestPolyCoeff())

    # calculate radius of lane
    leftRadiusMeter, rightRadiusMeter = calcRadius(left_line_x, right_line_x, both_lines_y)
    leftLine.setRadiusOfCurvature(leftRadiusMeter)
    rightLine.setRadiusOfCurvature(rightRadiusMeter)
    
    # draw the polyfitted lines on undistorted original image
    polyfit_on_undistorted = drawPolyfitOnImage(rgb_undistort, binary_combined_warped, Minv, left_line_x, right_line_x, both_lines_y)
    imageBank[11] = polyfit_on_undistorted
    if retNr is 11:
        return polyfit_on_undistorted
    
    # calc the deviation of the lane center of vehicle
    vehicleCenterDeviation = calcVehicleDeviation(left_line_x, right_line_x, both_lines_y)
    leftLine.setLineBasePos(vehicleCenterDeviation)
    rightLine.setLineBasePos(vehicleCenterDeviation)

    # write text on image
    resultImage = writeText(polyfit_on_undistorted, (leftRadiusMeter+rightRadiusMeter)/2, vehicleCenterDeviation, widthMeter, correctionStatement)
    imageBank[12] = resultImage
    if retNr is 12:
        return resultImage, leftLine, rightLine
    
    if format == 'collage4':
        return genCollage(4, imageBank), leftLine, rightLine
    elif format == 'collage9':
        return genCollage(9, imageBank), leftLine, rightLine
    
    
    return resultImage, leftLine, rightLine

def genCollage(amount, imageBank):
    '''
        generating a 2x2 or 3x3 collage
        amount: 4 -> 2x2 collage; 9 _> 3x3 collage
        return: imageCollage
    '''
    resultImage = None
    

    if amount == 4:
        row1 = cv2.hconcat((imageBank[1], imageBank[5]))
        row2 = cv2.hconcat((imageBank[10], imageBank[12]))
        resultImage = cv2.vconcat((row1, row2))
        resultImage = cv2.resize(resultImage, (1920, int((1920/resultImage.shape[1]) * resultImage.shape[0])))
    
    elif amount == 9:
        row1 = cv2.hconcat((imageBank[1], imageBank[2], imageBank[4]))
        row2 = cv2.hconcat((imageBank[5], imageBank[6], imageBank[8]))
        row3 = cv2.hconcat((imageBank[9], imageBank[10], imageBank[12]))
        resultImage = cv2.vconcat((row1, row2, row3))
        resultImage = cv2.resize(resultImage, (1920, int((1920/resultImage.shape[1]) * resultImage.shape[0])))
    
    return resultImage

def calcVehicleDeviation(left_poly_x, right_poly_x, poly_y):
    '''
        calculating the deviation of vehicle of the center of the lane
        leftx: x-values for the left line
        rightx: x-values for the right line
        return: deviationMeters (neg => left of center, pos => right of center)
    '''
    
    # the distance of the left and right polyline in pixels
    lineDistancePixels = right_poly_x[-1] - left_poly_x[-1]
    #print('lineDistancePixels', lineDistancePixels)
    
    # the center of the lane in Pixels
    laneCenterAtPixel = left_poly_x[-1] + lineDistancePixels/2
    #print('laneCenterAtPixel', laneCenterAtPixel)
    
    # the position of the vehicle (camera) in pixels
    cameraCenterOfImg = 1280/2
    #print('cameraCenterOfImg', cameraCenterOfImg)
    
    # deviation of the vehicle from the lane center in pixels
    deviationInPixels = cameraCenterOfImg - laneCenterAtPixel
    #print('deviationInPixels', deviationInPixels)
    
    # if lane is 3.7 wide, how many meter is 1 pixel
    metersPerPixel = 3.7 / lineDistancePixels
    #print('metersPerPixel', metersPerPixel)
    
    # the deviation in meters
    deviationOfVehicleFromLaneCenterMeter = metersPerPixel * deviationInPixels
    #print('deviationOfVehicleFromLaneCenterMeter', deviationOfVehicleFromLaneCenterMeter)
    
    return deviationOfVehicleFromLaneCenterMeter

def calcLaneWidth(leftLine, rightLine):
    '''
        calculate lane width in meters
        leftLine: data of left line
        rightLine: data of right line
        return: lane width
    '''
    if leftLine.getX() and rightLine.getX():
        
#        x_px_per_meter = 700/3.7
#        laneWidthMeter = 3.7
        x_meter_per_px = 3.7/700
        actualLaneWidthPixel = rightLine.getX() - leftLine.getX()
        return actualLaneWidthPixel * x_meter_per_px
    
    return False

def isLaneWidthPlausible(widthMeter):
    '''
        determine whether lane width is plausible.
        widthMeter: width in meter
        return: bool if lane width is plausible
    '''
    result = False
    
    lowerBoundMeter = 3
    upperBoundMeter = 4.5
    
    if not widthMeter:
        return False
    
    if widthMeter < lowerBoundMeter or upperBoundMeter > upperBoundMeter:
        result = False
    else:
        result = True
    
#    print('isLaneWidthPlausible:', result)
    
    return result
    
def writeText(img, curvatureMeter, vehicleCenterDeviation, laneWidth, correctionStatement):
    '''
        writes the lane curvature onto the image
        img: image
        curvatureMeter: the curvature in meters
        return: image_with_text
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    dir = 'right'
    
    if vehicleCenterDeviation < 0:
        dir = 'left'
    
    cv2.putText(img, 'Radius of Curvature = '+str(int(curvatureMeter))+'m', (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Vehicle is '+'{:4.2f}'.format(abs(vehicleCenterDeviation))+'m '+dir+' of center', (50, 80), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(img, correctionStatement, (50, 110), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
#     if laneWidth != None:
#         cv2.putText(img, 'Lane Width is '+'{:4.2f}'.format(laneWidth)+'m ', (50, 140), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
   
    return img
    

def calcRadius(leftx, rightx, ploty):
    '''
        calculate the radius of the left and the right line
        leftx: x-values for the left line
        rightx: x-values for the right line
        ploty: y-values for both lines
        return: leftRadius, rightRadius
    '''
    # Generate some fake data to represent lane-line pixels
#    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
#    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
#    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
#                                  for y in ploty])
#    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
#                                    for y in ploty])
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad
    

def drawPolyfitOnImage(undistorted, warped, Minv, left_fitx, right_fitx, ploty):
    '''
        draws the polyfit lines as a binary image
        sampleOneChannelImage: to sample the shape from for the target image
        left_poly_x: x-values for the left line
        right_poly_x: x-values for the right line
        poly_y: y-values for both lines
        return: binary mask of the polyfit lines
    '''
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    return result

def drawPolyfit(sampleOneChannelImage, left_poly_x, right_poly_x, poly_y):
    '''
        draws the polyfit lines as a binary image
        sampleOneChannelImage: to sample the shape from for the target image
        left_poly_x: x-values for the left line
        right_poly_x: x-values for the right line
        poly_y: y-values for both lines
        return: binary mask of the polyfit lines
    '''
    
    
    # create an empty image with the same spacial shape of the source image, but with 1 color channel
    polyfit_binary = np.zeros_like(sampleOneChannelImage)
 
    # create binary    
    for i in range(len(poly_y)):
        for x_between in range(int(left_poly_x[i]), int(right_poly_x[i])):
         
            polyfit_binary[int(poly_y[i])][int(left_poly_x[i]):int(right_poly_x[i])] = 1
 
    
    return polyfit_binary

def generateLineXYValues(sampleImage, left_poly_coeff, right_poly_coeff):
    '''
        generate x and y values from polyfit coefficients
        sampleImage: sample image of right shape
        left_poly_coeff: polyfit coefficients for the left line
        right_poly_coeff: polyfit coefficients for the right line
        return: left_x, right_x, plot_y
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, sampleImage.shape[0]-1, sampleImage.shape[0] )
    left_fitx = left_poly_coeff[0]*ploty**2 + left_poly_coeff[1]*ploty + left_poly_coeff[2]
    right_fitx = right_poly_coeff[0]*ploty**2 + right_poly_coeff[1]*ploty + right_poly_coeff[2]

    return left_fitx, right_fitx, ploty


def smoothPolyCoeff(leftLine, rightLine, left_poly_coeff, right_poly_coeff):
    '''
        if one polyfit jumps and the other remains pretty much the same as in the last timestep
        the jumping one will be substituted by a parallel copy of the steady one
    '''
    
    statement = "no correction"

    if (len(leftLine.getRecentPolyCoeff()) > 0):
#         print('full coeffs:', left_poly_coeff)
#         print('3rd coeffs:', left_poly_coeff[2])
#         print('full recent coeffs:', leftLine.getRecentPolyCoeff()[-1])
#         print('3rd coeffs:', leftLine.getRecentPolyCoeff()[-1][2])
        
        leftChangePx = left_poly_coeff[2] - leftLine.getRecentPolyCoeff()[-1][2]
        rightChangePx = right_poly_coeff[2] - rightLine.getRecentPolyCoeff()[-1][2]
    
#         print('change of left line:', leftChangePx)
#         print('change of right line:', rightChangePx)
    
    
        # if the lines are diverging
        if leftChangePx - rightChangePx > 100:
#             print('ITs A JUMP!')
            
            # the line with the biggest change is considered faulty
            if abs(leftChangePx) > abs(rightChangePx):
                # left line faulty
                # overwrite the faulty left poly coeffs with the poly coeffs of the right
#                 print('left is faulty')
                left_poly_coeff = leftLine.getBestPolyCoeff()
                
                # get the 3rd coeff of last frame
#                left_poly_coeff[2] = leftLine.getRecentPolyCoeff()[-1][2]
                # overwrite the 1st and 2nd coeff with the values of the right line
#                left_poly_coeff[0] = right_poly_coeff[0]
#                left_poly_coeff[1] = right_poly_coeff[1]
                statement = "left is faulty - will be corrected"

            else:
                # right is faulty
#                 print('right is faulty')
                right_poly_coeff = rightLine.getBestPolyCoeff()
                # get the coeffs of last frame
#                right_poly_coeff[2] = rightLine.getRecentPolyCoeff()[-1][2]
                # overwrite the 1st and 2nd coeff with the values of the left line
#                right_poly_coeff[0] = left_poly_coeff[0]
#                right_poly_coeff[1] = left_poly_coeff[1]
                statement = "right is faulty - will be corrected"
        
    return left_poly_coeff, right_poly_coeff, statement

def findLinesSimple(binary_warped, left_fit, right_fit):
    
    '''
        searches for lines near the region where it has found lines in last frame
        binary_warped: binary image from birds eye view
        left_fit: polynomial coefficients of the left line
        right_fit: polynomial coefficients of the right line
        return: left_line_x, right_line_x, both_lines_y
    '''
 
     # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    for i in range(0, ploty):
        out_img[ploty[i], left_fitx[i]:right_fitx[i]] = [0, 0, 255]

    return out_img, left_fit, right_fit

def findLines(binary_warped):
    
    '''
        searches for lines
        binary_warped: binary image from birds eye view
        return: image_with_sliding_windows, polyfit_left_x, polyfit_right_x, polyfit_y
    '''
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
         
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # visualize it
#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
#     plt.clf()
#     plt.imshow(out_img)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)

    return out_img, left_fit, right_fit

def transformToBirdsView(img):
    '''
        transforms an image from a front facing camera to birds-eye-view
        the transformation is made by a fixed source -> destination mapping that has been measured from sample photos with straight lane lines
        img: input image
        return warped image
    '''
    
    # define source points
    src_xindent_lower = 200
    src_xindent_upper = 595
    src_yindent_upper = 450

    # define source points for transformation
    src = np.float32( [[ src_xindent_lower,                 img.shape[0] ],    # left lower corner
                        [ img.shape[1]-src_xindent_lower,   img.shape[0] ],     # right lower corner
                        [ img.shape[1]-src_xindent_upper,   src_yindent_upper ],    # right upper corner
                        [ src_xindent_upper,                src_yindent_upper ] ] ) # left upper corner
    
    # define destination points
    dst_xindent_lower = 250

    # define destination points for transformation
    dst = np.float32( [[ dst_xindent_lower,                img.shape[0] ],     # left lower corner
                        [ img.shape[1]-dst_xindent_lower,   img.shape[0] ],# right lower corner
                        [ img.shape[1]-dst_xindent_lower,   0 ],           # right upper corner
                        [ dst_xindent_lower,                0 ] ] )        # left upper corner

    # visualize source points
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    verts = np.copy(src)
#    print(verts.shape)
    verts = np.vstack([verts, verts[0]])
    
    codes = [ Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY,
            ]
    
    path = Path(verts, codes)
    
    patch = patches.PathPatch(path, edgecolor='r', facecolor='none', lw=2)
    
    ax.add_patch(patch)
#    plt.show()

    
    # create transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    img_size = (img.shape[1], img.shape[0])
    
    # transformation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    # show result
    fig2, ax2 = plt.subplots(1)
    ax2.imshow(warped)

    # plot dst points on image
    verts2 = np.copy(dst)
#    print(verts2.shape)
    verts2 = np.vstack([verts2, verts2[0]])
    
    path = Path(verts2, codes)
    
    patch = patches.PathPatch(path, edgecolor='r', facecolor='none', lw=2)
    
    ax2.add_patch(patch)
#    plt.show()
    
    plt.clf()

    return warped, [fig, fig2], M, Minv
    

def combineBinaries(listBinaries):
    '''
        combines 2 binaries to a single one
        listBinaries: list of 2 binaries
    '''
    # Stack each channel to view their individual contributions in green and blue respectively
    color_binary = np.dstack(( np.zeros_like(listBinaries[0]), listBinaries[0], listBinaries[1]))
    
    combined_binary = np.zeros_like(listBinaries[0])
    combined_binary[(listBinaries[0] == 1) | (listBinaries[1] == 1)] = 1
    
    return combined_binary


def getBinarySHls(rgb, s_thresh):
    '''
        isolates the s channel of HLS colorspace and creates a thresholded binary
        rgb: input image in RGB colorspace
        s_thresh: tuple of min and max threshold for the binary generation
        return: binary image of the thresholded s channel of an image in HLS colorspace
    '''
    
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    
    binary_output = np.zeros_like(S)
    binary_output[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1
    
    # 3) Return a binary image of threshold result
    return binary_output
    

def getBinaryMagSobelXY(gray, sobel_kernel, mag_sobelxy_thresh):
    '''
        calculates the magnitude of sobel and creates a thresholded binary
        gray: input image in grayscale
        sobel_kernel: size of the sobel kernel
        mag_sobelxy_thresh: tuple of min and max threshold for the binary generation
        return: binary image of the thresholded magnitude sobelxy operator
    '''

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_abs_sobelxy = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_abs_sobelxy)
    binary_output[(scaled_abs_sobelxy >= mag_sobelxy_thresh[0]) & (scaled_abs_sobelxy <= mag_sobelxy_thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

def calibrateCamera(calDir):
    '''
        function to calibrate camera with several images of a chessboard 9x6, taken with that camera
        calDir: directory with camera calibration images
        return[0] ret: True if calibration was successful
        return[1] mtx: calibration matrix
        return[2] dist: distortion coefficients
        return[3] rvecs: rotation vectors for camera position in the world
        return[4] tvecs: translation vectors for camera position in the world
    '''
    
    # calibration save file
    calibrationPkl = calDir + '/.calibration.pkl'
    
    # if one exists, then the calibration can be loaded from there instead of new calculation
    if os.path.isfile(calibrationPkl):
        log('info', 'precalculated calibration file found - loading that: '+calibrationPkl)
        [ret, mtx, dist, rvecs, tvecs] = pickle.load(open(calibrationPkl, "rb"))
        return ret, mtx, dist, rvecs, tvecs
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # for 9 * 6 corner points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    fnameImages = glob.glob(calDir + '/*')
    
    # for every image
    # collect the image points and the object points (these are the same for every image)
    for fname in fnameImages:
        
        # read image
        img = cv2.imread(fname)
    
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            log('info', 'chess board corners found in image '+fname.split('/')[-1])
            
            # draw the found corners and display them
            #img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            
            # show to check
            #plt.imshow(img)
            #plt.show()
            #sys.exit(0)
            
            objpoints.append(objp)
            imgpoints.append(corners)
    
        else:
            log('warn', 'skipping - chess board corners NOT found in image '+fname.split('/')[-1])
    
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # write calibration as file pkl to avoid next time calculation
    log('info', "writing camera calibration to pickle " + calibrationPkl)
    pickle.dump( [ret, mtx, dist, rvecs, tvecs], open(calibrationPkl, "wb") )


    # return
    return ret, mtx, dist, rvecs, tvecs
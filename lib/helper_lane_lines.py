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

def laneLinePipeline(rgb, mtx, dist, outDir, retNr, sobel_kernel=5, mag_sobelxy_thresh=(30, 100), hls_thresh=(170, 255)):
    '''
        processes an image from input to output in finding a lane line
        img: input image in bgr colorspace
        mtx: camera calibration matrix
        dist: camera distortion coefficients
        outDir: output directory path
        visLog: if True, also output the single steps in pipeline
        sobel_kernel: size of the sobel kernel
        mag_sobelxy_thresh: tuple of min and max threshold for the binary generation
        return: image with detected-lanes-overlay
    '''

    if retNr == 0:
        return rgb

    # undistort
    #undistort = np.copy(rgb)
#    log('debug', 'undistort image')
    rgb_undistort = cv2.undistort(rgb, mtx, dist, None, mtx)
    if retNr == 1:
        return rgb_undistort

    # convert to grayscale
#    log('debug', 'convert to grayscale')
    gray = cv2.cvtColor(rgb_undistort, cv2.COLOR_RGB2GRAY)
    if retNr == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # create binary mask of the thresholded magnitude of sobel operator
#    log('debug', 'create binary mask with abs_sobelxy operator')
    binary_output_abs_sobelxy = getBinaryMagSobelXY(gray, sobel_kernel, mag_sobelxy_thresh)
    if retNr == 3:
        tmp = binary_output_abs_sobelxy * 255
        return cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    
    # create binary mask of the thresholded s of the colorspace hls
#    log('debug', 'create binary mask with the s of the HLS color space')
    binary_output_s_of_hls = getBinarySHls(rgb_undistort, hls_thresh)
    if retNr == 4:
        tmp = binary_output_s_of_hls * 255
        return cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    
    # create a combined binary (gradient and colorspace)
#    log('debug', 'combine sobel and s binary mask to a single binary mask')
    binary_combined = combineBinaries([binary_output_abs_sobelxy, binary_output_s_of_hls])
    if retNr == 5:
        tmp = binary_combined * 255
        return cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # warp image from camera view to birds eye view
#    log('debug', 'transform image from camera view to birds eye view')
    binary_combined_warped, figs = transformToBirdsView(binary_combined)
    if retNr == 6:
        figs[0].canvas.draw() # draw the canvas, cache the renderer
        tmp = np.fromstring(figs[0].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tmp2 = tmp.reshape(figs[0].canvas.get_width_height()[::-1] + (3,))
        return tmp2
    if retNr == 7:
        figs[1].canvas.draw() # draw the canvas, cache the renderer
        tmp = np.fromstring(figs[1].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tmp2 = tmp.reshape(figs[1].canvas.get_width_height()[::-1] + (3,))
        return tmp2
    if retNr == 8:
        tmp = binary_combined_warped * 255
        return cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    
    # take a histogram along all the columns in the lower half of the image
    histogram = np.sum(binary_combined_warped[binary_combined_warped.shape[0]//2:, :], axis=0)
    if retNr == 9:
        plt.clf()
        fig = plt.figure(1)
        ax = plt.axes()
        plt.plot(histogram)
        fig.canvas.draw()
        tmp = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tmp2 = tmp.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return tmp2
    
    # finding the lines
    found_lines = findLines(binary_combined_warped)
    if retNr == 10:
        return found_lines
    
    return found_lines

def findLines(binary_warped):
    
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
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

#    plt.show()
    return out_img

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
    dst_xindent_lower = 300

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
    

    return warped, [fig, fig2]
    

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
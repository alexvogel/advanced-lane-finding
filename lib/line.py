from collections import deque
import numpy as np

# a class for the characteristics of each line detection
class Line():
    def __init__(self):
        # n
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False  

        # x value of the lowest line position
        self.x = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        # x values of the last n fits of the line
        self.recent_xfitted = deque([]) 
        #the lower edge line x values of all frames
        self.xhistory = deque([])

        #polynomial coefficients averaged over the last n iterations
        self.best_poly_coeff = None  
        #polynomial coefficients for the most recent fit
        self.current_poly_coeff = None 
        #polynomial coefficients of the last n fits
        self.recent_poly_coeff = []

        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

# SETTER
    def setAll(self, boolDetected, fitx, poly_coeff):
        self.setDetected(boolDetected)
        self.addRecentXfitted(fitx)
        self.setCurrentFit(poly_coeff)

    def setDetected(self, boolDetected):
        self.detected = boolDetected
    
#     def setRecentXFit(self, fitx):
#         self.addRecentXFitted(fitx)
#         self.addHistoryXFitted(fitx)
#     
#     def addRecentXfitted(self, fitx):
#         self.recent_xfitted.append(fitx)
#         if len(self.recent_xfitted) > self.n:
#             self.recent_xfitted.popleft()
#         
#        self.bestx = sum(self.recent_xfitted)/float(len(self.recent_xfitted))

    def addHistoryXfitted(self, fitx):
        self.xhistory.append(fitx)

    def setCurrentPolyCoeff(self, poly_coeff):
        self.current_poly_coeff = poly_coeff
        
        self.recent_poly_coeff.append(poly_coeff)
        if len(self.recent_poly_coeff) > self.n:
            self.recent_poly_coeff.pop()
        
        if len(self.recent_poly_coeff) > 1:
            # calc the average poly_coeff
            self.best_poly_coeff = sum(self.recent_poly_coeff) / len(self.recent_poly_coeff)
        
        else:
            self.best_poly_coeff = poly_coeff
        
    def setRadiusOfCurvature(self, radiusMeter):
        self.radius_of_curvature = radiusMeter
        
    def setLineBasePos(self, centerDeviationMeter):
        self.line_base_pos = centerDeviationMeter
        
# GETTER
    def isDetected(self):
        return self.detected
    
    def getBestPolyCoeff(self):
        return self.best_poly_coeff

    def getRecentPolyCoeff(self):
        return self.recent_poly_coeff

    def getX(self):
        return self.x

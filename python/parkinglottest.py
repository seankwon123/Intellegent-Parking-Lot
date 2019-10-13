# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:57:00 2019

@author: seank
"""
#from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import badautils
#from Rectangle import *
from firebase import Firebase

class DetectedBox :
    def __init__(self, x, y, width, height, confidence, classID):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.classID = classID
        
class CarTracker :
    def __init__(self, tracker, classID):
        self.tracker = tracker   
        self.classID = classID
        self.history = []

class ParkingLot:
    def __init__(self):
        self.config = {
                          "apiKey": "AIzaSyDNTUoTW_bAvGvHau-sql3JdhQKf2AIf6U",
                          "authDomain": "parking-e9aeb.firebaseapp.com",
                          "databaseURL": "https://parking-e9aeb.firebaseio.com",
                          "storageBucket": "",
                           "appId": "1:813778226838:web:8e5c5532ebf586e8"
                        }
        self.firebase = Firebase(self.config)
        self.db = self.firebase.database()
        
        self.resizeImageRatio = 1 #.5
        self.overlapThreshold = .7
        ## for overpass video
        self.trackingRegionInRealSize = badautils.Rectangle(260, 360, 1051, 460)
        
        ## for civicIn video
#        self.trackingRegionInRealSize = badautils.Rectangle(600, 400, 1600, 500)
        
        self.args = None
        self.numCars = 0
        self.trackers = []
        self.trackerID = 0
        
        self.COLORS = None
        self.LABELS = None
        self.net = None

        #(x1, y1, x2, y2)
        self.trackingRegion = None
        # will be set with resizeRatio
        self.imageWidth = 0
        self.imageHeight = 0
        
        self.ln = None
        self.start = None
        self.end = None
        
        self.vs = None
        self.writer = None
        
        self.total = None
       
        
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

    def initArgs(self):    
        #construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", help="path to input video")
        ap.add_argument("-o", "--output", help="path to output video")
        ap.add_argument("-t", "--tracker", type=str, default="csrt",
                        help="OpenCV object tracker type")
        ap.add_argument("-y", "--yolo", required=True, 
                        help="base path to YOLO directory")
        ap.add_argument("-c", "--confidence", type=float, default=0.5, 
                        help="minimum probability to filter weak detections")
        ap.add_argument("-s", "--threshold", type=float, default=0.3,
                        help="threshold when applyong non-maxima suppression")
        ap.add_argument("-b", "--buffer", type=int, default=64,
                        help="max buffer size")
        self.args = vars(ap.parse_args())
 

    def initYOLO(self):

        #load the COCO class labels 
        labelsPath = os.path.sep.join([self.args["yolo"], "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")
         
        # initialize a list of colors to represent each possible class
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
         
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([self.args["yolo"], "yolov3.weights"])
        configPath = os.path.sep.join([self.args["yolo"], "yolov3.cfg"])
         
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        self.net = net
        self.ln = ln
        
    def initVideo(self):    
        #initialize the video stream, pointer to output vid file and frame dimensions
        if not self.args.get("input", False):
            print("[INFO] starting video stream...")
            self.vs = cv2.VideoCapture(1)
            time.sleep(1.0)
            
        else:
            self.vs = cv2.VideoCapture(self.args["input"])
            time.sleep(1.0)
      
                  
        retImg, image = self.vs.read()            
        if retImg == False:
            return False
        self.imageWidth = int(image.shape[1] * self.resizeImageRatio)
        self.imageHeight = int(image.shape[0] * self.resizeImageRatio)
        
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(self.args["output"], fourcc, 30, 
                                (self.imageWidth, self.imageHeight), True)
        
        #try to determine the tot number of frames in vid file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            self.total = int(self.vs.get(prop))
            print("[INFO] {} total frames in video".format(self.total))
        
        #an error occurred while trying to determine total frames?
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            self.total = -1

        return True
        

    def detectCars(self, image):
        
# =============================================================================
#         if self.imageWidth is None or self.imageHeight is None:
#             (self.imageHeight, self.imageWidth) = image.shape[:2]
# =============================================================================
    
        #contruct blob from input frame then perform forward pass of YOLO object
        # detector, giving us our bounding boxes and probabilities
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 316), swapRB=True, crop=False)
        self.net.setInput(blob)
        self.start = time.time()
        layerOutputs = self.net.forward(self.ln)
        self.end = time.time()
        
        #initialize lists of detected bounding boxes, confidences, and class IDs
        yoloBoxes = []
        confidences = []
        classIDs = []
        
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
  
       			#filter out weak predictions by ensuring the detected
    			   # prob > minimum probability
    
                if confidence > self.args["confidence"]:
    				    #scale the bounding box coordinates back relative to
    			     	 # the size of the image, keeping in mind that YOLO
    				    # actually returns the center (x, y) of the bounding box
                    # followed by the boxes' width and height
                    box = detection[0:4] * (np.array([self.imageWidth, self.imageHeight, self.imageWidth, self.imageHeight]))
                    (centerX, centerY, width, height) = box.astype("int")
    				    #use the center to derive the top and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                                        
                    
    				    #update our list of bounding box coordinates, confidences, and IDs
                    yoloBoxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)    
                    
        #apply non-maxima suppression to suppress weak, overlapping boxes           
        idxs = cv2.dnn.NMSBoxes(yoloBoxes, confidences, self.args["confidence"], self.args["threshold"])
    
        detectedBoxes = []
        #make sure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (yoloBoxes[i][0], yoloBoxes[i][1])
                (w, h) = (yoloBoxes[i][2], yoloBoxes[i][3])
                
                # save the true detected boxes
                if self.LABELS[classIDs[i]] == "car" or self.LABELS[classIDs[i]] == "truck":
                    detectedBoxes.append(DetectedBox(x, y, w, h, confidences[i], classIDs[i]))
                    
                    #draw a box rectangle and label on the frame
                    color = [int(c) for c in self.COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                               
        return detectedBoxes
    
    def isInTrackingRegion(self, dx, dy):
        if dx < self.trackingRegion.x1 or dx > self.trackingRegion.x2 \
            or dy < self.trackingRegion.y1 or dy > self.trackingRegion.y2:
            return False
        return True
    
    def isRectOverlap(self, box1, box2):
        overlapBox = (box1 & box2)
        if overlapBox and overlapBox.area() > 0:
            return True
        return False
    #image    
    # 0: right-top
    # 1: left-top
    # 2: left-bottom
    # 3: right-bottom        
    def getDirection(self, history):
        totalX = 0
        totalY = 0
        i = 0
        while i < len(history) - 1:
            totalX += history[i + 1][0] - history[i][0]
            totalY += history[i + 1][1] - history[i][1]
            i += 1
        #oversimplified conditions
        if totalY <= 0 and totalX >= 0:
            return 0
        elif totalY <= 0 and totalX <= 0:
            return 1
        elif totalY >= 0 and totalX <= 0:
            return 2
        else:
            return 3         
        
    def isLeavingOrComing(self, history):
        if self.getDirection(history) == 0 or self.getDirection(history) == 1:
            self.numCars -= 1
        if self.getDirection(history) == 2 or self.getDirection(history) == 3:
            self.numCars += 1
        data = {"cars": self.numCars}
        self.db.update(data)    
        
            
    def trackCars(self, image, detectedBoxes):
        #trackingOK = True
        #boxIdx = 0
        #while boxIdx < len(self.trackers):
        for i, tracker in enumerate(self.trackers):
            (success, box) = tracker.tracker.update(image)
            tracker.history.append(box)
            
            if success is False:
                print("***** Failed to trackers update")
                del self.trackers[i]
                
            
        if self.trackers is None and detectedBoxes is None:
            return
        
        
        for detectedBox in detectedBoxes:
            dx, dy, dw, dh = int(detectedBox.x), int(detectedBox.y), \
                                int(detectedBox.width), int(detectedBox.height)
            
            # check boundaries
            #if not self.isInTrackingRegion(dx + int(dw/2), dy + int(dh/2)):  
            #box1 = badautils.Rectangle(dx, dy, dx + dw, dy + dh)
            #box2 = self.trackingRegion
            #if not self.isRectOverlap(box1, box2):
              #  continue
            if not self.isInTrackingRegion(dx + int(dw/2), dy + int(dh/2)):
                continue
            
            
            needToBeTracked = True
            for i, tracker in enumerate(self.trackers):
                if len(tracker.history) < 1 :
                    continue
                (tx, ty, tw, th) = [int(v) for v in tracker.history[-1]] #box extracted from (success, box)
                
                #if not self.isInTrackingRegion(tx + int(tw/2), ty + int(th/2)):  
                
                tbox = badautils.Rectangle(tx, ty, tx + tw, ty + th)
                if not self.isRectOverlap(tbox, self.trackingRegion):
                    self.isLeavingOrComing(tracker.history)
                    del self.trackers[i]
                    continue
                
                cv2.rectangle(image, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 1)                        
                box1 = badautils.Rectangle(tx, ty, tx + tw, ty + th)
                box2 = badautils.Rectangle(dx, dy, dx + dw, dy + dh)
                overlapBox = (box1&box2)

#                cv2.rectangle(image, (dx, dy), (dx + dw, dy + dh), (255, 0, 0), 1)

                if overlapBox:

                    cv2.rectangle(image, (overlapBox.x1, overlapBox.y1), (overlapBox.x2, overlapBox.y2), (255, 0, 255), 1)
                    
                    if box1.area() > box2.area():
                        overlapPercent = overlapBox.area() / box2.area()
                    else:
                        overlapPercent = overlapBox.area() / box1.area()

                        
                    #overlapPercent = overlapBox.area() / (box1.area() + box2.area() - overlapBox.area())
                    #make sure use area as a percentage
                    if overlapPercent > self.overlapThreshold:
                        needToBeTracked = False
                        break

            if needToBeTracked:
                tracker = self.OPENCV_OBJECT_TRACKERS[self.args["tracker"]]()
                tracker.init(image, (dx, dy, dw, dh))
                newTrackCar = CarTracker(tracker, self.trackerID)
                self.trackers.append(newTrackCar)
                self.trackerID += 1
                
        # check the direction of moving object when trackers remain
        for i, tracker in enumerate(self.trackers):
            if len(tracker.history) < 1 :
                continue
            (tx, ty, tw, th) = [int(v) for v in tracker.history[-1]] #box extracted from (success, box)
            
            if not self.isInTrackingRegion(tx + int(tw/2), ty + int(th/2)):  
                self.isLeavingOrComing(tracker.history)
                del self.trackers[i]
                continue

        

    def displayOutput(self, image):
        # display 
        if self.total > 0:
            elap = (self.end - self.start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                     self.total))

        text = "# of Trackers: {}".format(len(self.trackers))
        cv2.putText(image, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)   
        text2 = "# of cars: {}".format(self.numCars)
        cv2.putText(image, text2, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        cv2.imshow("tracking + yolo", image)
        key = cv2.waitKey(1) & 0xFF
        
        print(self.numCars)
          
        if key == ord("q"):
            return False
        return True

    def detectAndTrack(self):
        #loop over frames from vid file
        x1, y1, x2, y2 = int(self.resizeImageRatio * self.trackingRegionInRealSize.x1), \
                         int(self.resizeImageRatio * self.trackingRegionInRealSize.y1), \
                         int(self.resizeImageRatio * self.trackingRegionInRealSize.x2), \
                         int(self.resizeImageRatio * self.trackingRegionInRealSize.y2)
        self.trackingRegion = badautils.Rectangle(x1, y1, x2, y2)  

        while True:    
            retImg, image = self.vs.read()            
            if retImg == False:
                break;
            ## resize for debug ==> 50%
            image = cv2.resize(image, (self.imageWidth, self.imageHeight))
                     
            cv2.rectangle(image, (self.trackingRegion.x1, self.trackingRegion.y1), \
                          (self.trackingRegion.x2, self.trackingRegion.y2), (105, 105, 105), 2)
            
            detectedBoxes = self.detectCars(image)
            self.trackCars(image, detectedBoxes)
            if not self.displayOutput(image):
                break
            
            #not working because of the fact that we resized the image, so the height and width are no longer 
            #the same value as the original, so it fails the assertions needed to use "writer"
            self.writer.write(image)
        
def main():
    parkingLot = ParkingLot()
    parkingLot.initArgs()
    parkingLot.initYOLO()
    if parkingLot.initVideo():
                
        parkingLot.detectAndTrack()
        
        print("[INFO] cleaning up...")
    
        parkingLot.vs.release()
        
        parkingLot.writer.release()
        cv2.destroyAllWindows()
    else:
        print("{} is an invalid file".format(parkingLot.args["input"]))
        
if __name__== "__main__":
  main()
  
  
    
    
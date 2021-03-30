# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *
import sys
sys.path.insert(1,"/util")
from util.utilityFuncs import *
import os
import pandas as pd
from matplotlib import pyplot as plt

########## AMBULANCE EMERGENCY VEHICLE USING TEMPLATE DETECTION

def ambulanceDetection(frame):
    ambulanceCount = 0
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    template = cv2.imread("ambulance.jpeg" , 0)
    height, width = template.shape[::]
    res = cv2.matchTemplate(frame_gray , template , cv2.TM_CCOEFF_NORMED)
    cv2.imshow('frame',res)
    threshold = 0.8
    loc = np.where(res>=threshold)
    for pt in zip(loc[::-1]):
        ambulanceCount+=1
    return ambulanceCount
	
# Setting Parameters here
FRAMES_BEFORE_CURRENT = 10 
inputWidth, inputHeight = 416, 416

list_of_vehicles = predictionClasses()
# Initialise Pandas DataFrame
cols = ['Frame Number' , 'Vehicle Count' , 'Emergency Count']
df = pd.DataFrame(columns = cols)
# Get the info passed via command line
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()
def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)
# Set Colors for bounding boxes
COLORS = setColors(LABELS)

# Load the network from disk
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Name of last layers of NN
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Specifying coordinates for a default line 
x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
count_prev_frame_detections = [0 for i in range(FRAMES_BEFORE_CURRENT)]

num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
count = 0


# loop over frames from the video file stream
frame_num = list()
vehicle_detected_count = list()
emergency_count = list()

def appendToList(count,display_vehicle_count,ambulance_count):
    frame_num.append(count)
    vehicle_detected_count.append(display_vehicle_count)
    emergency_count.append(ambulance_count)
while True:
	
	num_frames+= 1
	count+=1
	print(count)
	if(count>500):
		break

	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 

	
	start_time, num_frames = displayFPS(start_time, num_frames)

	(grabbed, frame) = videoStream.read()
	if not grabbed:
		break

	ambulance_count = ambulanceDetection(frame)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	for output in layerOutputs:
	
		for i, detection in enumerate(output):
		
			scores = detection[5:]
	
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > preDefinedConfidence:
				
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
                            
			
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,preDefinedThreshold)
	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame , LABELS , COLORS)
	vehicle_count, current_detections,curr_detection_count = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame , LABELS , list_of_vehicles)
	display_vehicle_count = len(boxes) 
	displayVehicleCount(frame, display_vehicle_count)
	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
	previous_frame_detections.pop(0) 
	previous_frame_detections.append(current_detections) , appendToList(count,display_vehicle_count,ambulance_count)
    
	


# MAKE A CSV FILE OF OBSERVATIONS

df[cols[0]] = frame_num
df[cols[1]] = vehicle_detected_count
df[cols[2]] = emergency_count
df.to_csv("Vehicle_Detections_lane2.csv" , index = False)
print("[INFO] cleaning up...")
writer.release()

videoStream.release()

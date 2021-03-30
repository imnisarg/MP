# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
import os
def predictionClasses():
    predClasses = ["bicycle","car","motorbike","bus","truck", "train"]
    return predClasses


### SETTING PARAMETERS HERE

FRAMES_BEFORE_CURRENT = 10 
inputWidth, inputHeight = 416, 416
def setColors(LABELS):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS


def displayVehicleCount(frame, vehicle_count):
	cv2.putText(frame, 'Detected Vehicles: ' + str(vehicle_count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0xFF, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL,)

def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
	x1_line, y1_line, x2_line, y2_line = line_coordinates 

	if( (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and(y_mid_point >= y1_line and y_mid_point <= y2_line+5)):
		return True
	return False

def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		os.system('clear') 
		print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame , LABELS , COLORS):
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
			cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)



def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf 
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: 
			continue
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame , LABELS , list_of_vehicles):
	current_detections = {}
	curr_detection_count = 0
	print(list_of_vehicles)
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			centerX = x + (w//2)
			centerY = y+ (h//2)
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count 
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1
					curr_detection_count+=1
					
				ID = current_detections.get((centerX, centerY))
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 
					curr_detection_count+=1
				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, current_detections,curr_detection_count
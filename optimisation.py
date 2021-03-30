## GET THE DATA
import pandas as pd 
lane1 = pd.read_csv("Vehicle_Detections_lane1.csv")
lane2 = pd.read_csv("Vehicle_Detections_lane2.csv")
lane3 = pd.read_csv("Vehicle_Detections_lane3.csv")
lane4 = pd.read_csv("Vehicle_Detections_lane4.csv")
n1 = len(lane1['Vehicle Count'])
n1-=1
n2 = len(lane2['Vehicle Count'])
n2-=1
n3 = len(lane3['Vehicle Count'])
n3-=1
n4 = len(lane4['Vehicle Count'])
n4-=1
trafficFlow_lane1 = lane1['Vehicle Count'][n1] - lane1['Vehicle Count'][0]
trafficFlow_lane2 = lane2['Vehicle Count'][n2] - lane2['Vehicle Count'][0]
trafficFlow_lane3 = lane3['Vehicle Count'][n3] - lane3['Vehicle Count'][0]
trafficFlow_lane4 = lane4['Vehicle Count'][n4] - lane4['Vehicle Count'][0]

expectedTraffic_lane1 = trafficFlow_lane1*0.5 + lane1['Vehicle Count'][n1]
expectedTraffic_lane2 = trafficFlow_lane2*0.5 + lane2['Vehicle Count'][n2]
expectedTraffic_lane3 = trafficFlow_lane3*0.5 + lane3['Vehicle Count'][n3]
expectedTraffic_lane4 = trafficFlow_lane4*0.5 + lane4['Vehicle Count'][n4] 


emergency_lane1 = lane1['Emergency Count'][n1]

emergency_lane2 = lane2['Emergency Count'][n2]

emergency_lane3 = lane3['Emergency Count'][n3]

emergency_lane4 = lane4['Emergency Count'][n4]

expectedTraffic_lane1+=emergency_lane1*5
expectedTraffic_lane2+=emergency_lane2*5
expectedTraffic_lane3+=emergency_lane3*5
expectedTraffic_lane4+=emergency_lane4*5


lane1_time = (expectedTraffic_lane1/(expectedTraffic_lane1+expectedTraffic_lane2+expectedTraffic_lane3+expectedTraffic_lane4))*240

lane2_time = (expectedTraffic_lane2/(expectedTraffic_lane1+expectedTraffic_lane2+expectedTraffic_lane3+expectedTraffic_lane4))*240

lane3_time = (expectedTraffic_lane3/(expectedTraffic_lane1+expectedTraffic_lane2+expectedTraffic_lane3+expectedTraffic_lane4))*240

lane4_time = (expectedTraffic_lane4/(expectedTraffic_lane1+expectedTraffic_lane2+expectedTraffic_lane3+expectedTraffic_lane4))*240

min_time = min(min(lane1_time,lane2_time),min(lane3_time,lane4_time))
if(min_time<30):
    print("WE NEED ADJUSTMENTS")
    if(min_time==lane1_time):
        adjustmentTime = 30-min_time
        lane1_time = 30
        lane2_time-=adjustmentTime/3
        lane3_time-=adjustmentTime/3
        lane4_time-=adjustmentTime/3
    if(min_time==lane2_time):
        adjustmentTime = 30-min_time
        lane2_time = 30
        lane1_time-=adjustmentTime/3
        lane3_time-=adjustmentTime/3
        lane4_time-=adjustmentTime/3
    if(min_time==lane3_time):
        adjustmentTime = 30-min_time
        lane3_time = 30
        lane2_time-=adjustmentTime/3
        lane1_time-=adjustmentTime/3
        lane4_time-=adjustmentTime/3
    if(min_time==lane4_time):
        adjustmentTime = 30-min_time
        lane4_time = 30
        lane2_time-=adjustmentTime/3
        lane3_time-=adjustmentTime/3
        lane1_time-=adjustmentTime/3
    print("THE TIMMINGS ARE")
    print("lane1_time: ",lane1_time)
    print("lane2_time: ",lane2_time)
    print("lane3_time: ",lane3_time)
    print("lane4_time: ",lane4_time)
else:
    print("THE TIMMINGS ARE")
    print("lane1_time: ",lane1_time)
    print("lane2_time: ",lane2_time)
    print("lane3_time: ",lane3_time)
    print("lane4_time: ",lane4_time)
    
        
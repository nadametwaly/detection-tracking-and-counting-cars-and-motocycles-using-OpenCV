import cv2
from tracker import *

cap=cv2.VideoCapture("detection-tracking-and-counting-cars-motocycles/highway.mp4")


tracker = EuclideanDistTracker()
object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50)
while True:
    ret,frame=cap.read()
    height, width, _ =frame.shape
    
    ROI=frame[340:720, 500:800]
    mask=object_detector.apply(ROI)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>100:
            x,y,w,h=cv2.boundingRect(cnt)
            
            detections.append([x,y,w,h])
    
    boxes_ids=tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.putText(ROI,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        cv2.rectangle(ROI,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imshow("ROI",ROI)
        cv2.imshow("MASK",mask)
        cv2.imshow("frame",frame)
        
    if cv2.waitKey(40) == 27:
        break 
    
cv2.waitKey(0)
cv2.destroyAllWindows()
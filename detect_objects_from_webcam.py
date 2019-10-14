from imageai.Detection import ObjectDetection
import os
import cv2 
from PIL import Image

cap = cv2.VideoCapture(0) 

# Load Yolo
execution_path = os.getcwd()
detector = ObjectDetection()

#detector.setModelTypeAsTinyYOLOv3()  # YOLOv3
#detector.setModelPath( os.path.join(execution_path , "models/yolo-tiny.h5"))
detector.setModelTypeAsRetinaNet()  # Other types are TinyYOLOv3, YOLOv3
detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))  #yolo-tiny.h5

detector.loadModel() #detection_speed="fastest"

process_this_frame = True

while 1: 
    
	# reads frames from a camera 
    ret, img = cap.read() 
    # img_str = cv2.imencode('.jpg', img)[1].tostring()
    
    if process_this_frame:
		
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        
        detections = detector.detectObjectsFromImage(input_type="array", input_image=img, output_type="file")
        print(detections)
        # img_analyzed = Image.fromarray(detections, 'RGB')
        # print(img_analyzed)

    
        # Display the results
        for detection in detections:
            name = detection['name'] 
            location = detection['box_points']
            top = location[1]
            right = location[2]
            bottom = location[3]
            left = location[0]
            #print(name, ": ", top, ":", left)
     
            # Draw a box around the object identified
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_ITALIC
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', img)

    process_this_frame = not process_this_frame

    #print('*******************')

	# Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()

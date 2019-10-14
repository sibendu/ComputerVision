import numpy as np
import cv2 
import argparse

classes = "yolov3.txt"
weights = "yolov3.weights"
config =  "yolov3.cfg"

ap = argparse.ArgumentParser()
args = ap.parse_args()

cap = cv2.VideoCapture(0) 

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

process_this_frame = True

while 1: 
    
    ret, image = cap.read() 
    
    if process_this_frame:

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet(weights, config)
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        print(outs)

        # Display the resulting image
        #cv2.imshow('Video', img)

    process_this_frame = not process_this_frame

	# Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()

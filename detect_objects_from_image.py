from imageai.Detection import ObjectDetection
import os

# Load Yolo
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()  # Other types are TinyYOLOv3, YOLOv3
detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))  #yolo-tiny.h5
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="images/streetview.jpg", output_image_path="images/out.jpg")

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

print('*******************')
print('*******************')
print('*******************')

print(detections)

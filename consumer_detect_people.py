from flask import Flask, Response, render_template
from kafka import KafkaConsumer
import face_recognition
import cv2
import numpy as np
from imageai.Detection import ObjectDetection
import os

# Fire up the Kafka Consumer
topic = "my-video"
consumer = KafkaConsumer(topic, bootstrap_servers=['localhost:9092'])

# Set the consumer in a Flask App
app = Flask(__name__)

# Load a sample pictures and learn how to recognize it
obama_image = face_recognition.load_image_file("images/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# sibendu_image = face_recognition.load_image_file("images/sibendu.jpg")
# sibendu_face_encoding = face_recognition.face_encodings(sibendu_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
# 	sibendu_face_encoding,
    obama_face_encoding
]
known_face_names = [
#	"Sibendu",
    "Barack Obama"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    """
    This is the heart of our video display. Notice we set the mimetype to 
    multipart/x-mixed-replace. This tells Flask to replace any old images with 
    new values streaming through the pipeline.
    """
    return Response(
        get_video_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_video_stream():
    
    process_this_frame = True

    """
    Recieve streamed images from Kafka and convert 
    """
    for msg in consumer:

        if process_this_frame:

            img = msg.value 

            # ret, buffer = cv2.imencode('.jpg', frame)
            # buffer.tobytes())
		
            nparr = np.fromstring(img, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR )
		
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_ITALIC
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            img = cv2.imencode('.jpg', frame)[1].tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img + b'\r\n\r\n')
		
        process_this_frame = not process_this_frame

# start server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

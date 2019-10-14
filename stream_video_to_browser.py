from flask import Flask, request, Response, render_template
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    name = request.args.get('name')
    print(name)
    return render_template('index.html', name=name)

@app.route('/video_feed', methods=['GET'])
def video_feed():
    name = request.args.get('name')
    print(name)
    return Response(
        get_video_stream(name), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_video_stream(video_file):	

    video_file = "videos/"+video_file
    print(video_file) 

    video = cv2.VideoCapture(video_file)
    
    print('publishing video...')

    while(video.isOpened()):
        success, frame = video.read()

        # Ensure file was read successfully
        if not success:
            print("bad read!")
            break
        
        # Convert image to png
        ret, buffer = cv2.imencode('.jpg', frame)
		
        img = buffer.tobytes()
        #print(img)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img + b'\r\n\r\n')

    video.release()
    print('publish complete')

# start server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

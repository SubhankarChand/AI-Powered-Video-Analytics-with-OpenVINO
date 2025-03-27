from flask import Flask, render_template, Response
from flask_sock import Sock
import cv2
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

class VideoProcessor:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.visitor_count = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        logger.info("VideoProcessor initialized")

    def process_frame(self, frame):
        # FPS calculation
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # existing frame processing logic
        return frame

    def get_analytics(self):
        return {
            'total_visitors': self.visitor_count,
            'active_visitors': len(self.tracked_objects),
            'timestamp': datetime.now().isoformat(),
            'fps': self.current_fps
        }

video_processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@sock.route('/updates')
def handle_updates(ws):
    last_data = None
    while True:
        try:
            current_data = video_processor.get_analytics()
            if current_data != last_data:
                ws.send(json.dumps(current_data))
                last_data = current_data
            time.sleep(0.5) 
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            break

def generate_frames():
    while True:
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                logger.error("Failed to open camera")
                time.sleep(1)
                continue

            # Set properties one by one with checks
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed")
                    break

                try:
                    processed_frame = video_processor.process_frame(frame)
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    break

        except Exception as e:
            logger.error(f"Camera error: {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            time.sleep(1)

if __name__ == '__main__':
    try:
        logger.info("Starting server...")
        app.run(host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        logger.error(f"Server failed: {e}")
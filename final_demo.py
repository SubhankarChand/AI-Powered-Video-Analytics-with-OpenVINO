import cv2
import time
import logging
import numpy as np
import os
import csv
import io
from flask import Flask, render_template, Response, make_response
from flask_sock import Sock
import json
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from openvino.runtime import Core

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

class AllInOneProcessor:
    def __init__(self):
        # 1. LOAD PERSON DETECTOR
        try:
            model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")
            self.yolo_model = YOLO(model_path)
        except:
            self.yolo_model = None
            
        # 2. LOAD EMOTION DETECTOR
        self.core = Core()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_xml = os.path.join(current_dir, "models", "emotions-recognition-retail-0003.xml")
        model_bin = os.path.join(current_dir, "models", "emotions-recognition-retail-0003.bin")
        
        if os.path.exists(model_xml) and os.path.exists(model_bin):
            model = self.core.read_model(model_xml, model_bin)
            self.fer_model = self.core.compile_model(model, "CPU")
        else:
            self.fer_model = None

        self.emotion_labels = ["neutral", "happy", "sad", "surprise", "anger"]
        self.tracked_objects = {}
        self.next_id = 1
        self.visitor_count = 0
        
        # --- NEW: History Logger for CSV ---
        # Stores: { id: { 'entry_time': str, 'emotions': [] } }
        self.visit_history = {} 

    def process_frame(self, frame):
        if frame is None: return frame
        
        # 1. DETECT PEOPLE
        if self.yolo_model:
            results = self.yolo_model(frame, verbose=False, conf=0.25)
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append([x1, y1, x2, y2])
            self._update_tracking(detections, frame)

        return self._draw_ui(frame)

    def _update_tracking(self, detections, frame):
        active_ids = set()
        
        for box in detections:
            x1, y1, x2, y2 = box
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            
            best_id = None
            min_dist = 150 
            
            for tid, tdata in self.tracked_objects.items():
                lx, ly = tdata['center']
                dist = ((lx - center_x)**2 + (ly - center_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.visitor_count += 1
                self.tracked_objects[best_id] = {
                    'frames_unseen': 0, 
                    'emotion': deque(maxlen=10),
                    'current_emotion': "detecting..."
                }
                # --- NEW: Log entry time ---
                self.visit_history[best_id] = {
                    'entry_time': datetime.now().strftime("%H:%M:%S"),
                    'emotions': []
                }
            
            self.tracked_objects[best_id]['box'] = box
            self.tracked_objects[best_id]['center'] = (center_x, center_y)
            self.tracked_objects[best_id]['frames_unseen'] = 0
            active_ids.add(best_id)
            
            if self.fer_model:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    try:
                        resized = cv2.resize(face, (64, 64))
                        input_tensor = np.transpose(resized, (2, 0, 1))
                        input_tensor = np.expand_dims(input_tensor, 0).astype(np.float32)
                        res = self.fer_model([input_tensor])[self.fer_model.output(0)]
                        emotion = self.emotion_labels[np.argmax(res.flatten())]
                        self.tracked_objects[best_id]['emotion'].append(emotion)
                        
                        # Update live tracker
                        emotions = self.tracked_objects[best_id]['emotion']
                        if emotions:
                            dom = max(set(emotions), key=emotions.count)
                            self.tracked_objects[best_id]['current_emotion'] = dom
                            # --- NEW: Save to history ---
                            self.visit_history[best_id]['emotions'].append(dom)
                    except: pass

        cleanup = []
        for tid in self.tracked_objects:
            if tid not in active_ids:
                self.tracked_objects[tid]['frames_unseen'] += 1
                if self.tracked_objects[tid]['frames_unseen'] > 30:
                    cleanup.append(tid)
        for tid in cleanup: del self.tracked_objects[tid]

    def _draw_ui(self, frame):
        for tid, tdata in self.tracked_objects.items():
            if tdata['frames_unseen'] == 0:
                x1, y1, x2, y2 = tdata['box']
                emo = tdata['current_emotion']
                color = (0, 255, 200)
                t, d = 2, 20
                cv2.line(frame, (x1, y1), (x1+d, y1), color, t)
                cv2.line(frame, (x1, y1), (x1, y1+d), color, t)
                cv2.line(frame, (x2, y1), (x2-d, y1), color, t)
                cv2.line(frame, (x2, y1), (x2, y1+d), color, t)
                cv2.line(frame, (x1, y2), (x1+d, y2), color, t)
                cv2.line(frame, (x1, y2), (x1, y2-d), color, t)
                cv2.line(frame, (x2, y2), (x2-d, y2), color, t)
                cv2.line(frame, (x2, y2), (x2, y2-d), color, t)
                label = f"ID:{tid} | {emo.upper()}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-20), (x1+w+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        return frame
    
    def get_stats(self):
        active_list = []
        emotion_counts = defaultdict(int)
        for tid, tdata in self.tracked_objects.items():
            if tdata['frames_unseen'] < 5:
                emo = tdata['current_emotion']
                emotion_counts[emo] += 1
                active_list.append({'id': tid, 'emotion': emo, 'last_seen': "Now"})
        if not emotion_counts: emotion_counts['waiting'] = 1
        elif 'waiting' in emotion_counts and len(emotion_counts) > 1: del emotion_counts['waiting']

        return {
            'total_visitors': self.visitor_count,
            'active_visitors': len(active_list),
            'emotion_stats': dict(emotion_counts),
            'active_list': active_list,
            'timestamp': datetime.now().isoformat()
        }

    # --- NEW: Generate CSV Data ---
    def generate_report(self):
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Visitor ID', 'Entry Time', 'Dominant Emotion', 'All Emotions Detected'])
        
        for vid, data in self.visit_history.items():
            # Find most frequent emotion
            emos = data['emotions']
            dom = max(set(emos), key=emos.count) if emos else "Neutral"
            cw.writerow([vid, data['entry_time'], dom, ", ".join(emos[:5]) + "..."])
            
        return si.getvalue()

processor = AllInOneProcessor()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NEW: Download Route ---
@app.route('/download_report')
def download_report():
    csv_data = processor.generate_report()
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=visitor_report.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@sock.route('/updates')
def updates(ws):
    while True:
        try:
            ws.send(json.dumps(processor.get_stats()))
            time.sleep(0.5)
        except: break

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        frame = processor.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
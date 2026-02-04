import os
import cv2
import numpy as np
from openvino.runtime import Core
from collections import defaultdict, deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        # Initialize with proper model paths
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.core = Core()
        
        # Load models with validation
        self.yolo_model = self._load_model("yolov8n")
        self.fer_model = self._load_model("emotions-recognition-retail-0003")
        
        # Model configurations
        self.confidence_threshold = 0.6
        self.iou_threshold = 0.4
        self.class_names = ["person"]
        self.emotion_labels = ["neutral", "happy", "sad", "surprise", "anger"]
        
        # Tracking system
        self.tracked_objects = {}
        self.next_id = 0
        self.visitor_count = 0
        self.max_disappeared = 10
        self.emotion_history = defaultdict(lambda: deque(maxlen=20))
        
        logger.info("VideoProcessor initialized successfully")

    def _load_model(self, model_name):
        """Safe model loader with path validation"""
        model_path = os.path.join(self.model_dir, f"{model_name}.xml")
        weights_path = os.path.join(self.model_dir, f"{model_name}.bin")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        try:
            model = self.core.read_model(model_path, weights_path)
            return self.core.compile_model(model, "CPU")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            raise

    def process_frame(self, frame):
        try:
            # Validate input
            if frame is None or not isinstance(frame, np.ndarray):
                logger.warning("Invalid frame input")
                return frame
                
            # Processing pipeline
            detections = self._detect_objects(frame)
            self._update_tracks(detections, frame)
            return self._draw_analytics(frame)
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame

    def _detect_objects(self, frame):
        """Improved object detection with input validation"""
        try:
            input_tensor = self._preprocess_frame(frame)
            outputs = self.yolo_model([input_tensor])[self.yolo_model.output(0)]
            return self._postprocess_yolo(outputs, frame.shape)
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return []

    def _preprocess_frame(self, frame):
        """Standardized preprocessing"""
        input_shape = self.yolo_model.input(0).shape
        resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
        input_data = np.transpose(resized, (2, 0, 1))  # HWC to CHW
        return np.expand_dims(input_data, 0).astype(np.float32) / 255.0

    def _postprocess_yolo(self, outputs, frame_shape):
        """Robust postprocessing with boundary checks"""
        detections = []
        height, width = frame_shape[:2]
        
        try:
            outputs = np.transpose(outputs[0], (1, 0))  
            
            for output in outputs:
                scores = output[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence < self.confidence_threshold or class_id != 0:
                    continue
                    
                # Convert to image coordinates
                x, y, w, h = output[:4]
                x1 = max(0, int((x - w/2) * width))
                y1 = max(0, int((y - h/2) * height))
                x2 = min(width, int((x + w/2) * width))
                y2 = min(height, int((y + h/2) * height))
                
                if x1 >= x2 or y1 >= y2:  
                    continue
                    
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_name': "person"
                })
                
        except Exception as e:
            logger.error(f"Postprocessing error: {str(e)}")
            
        return detections

    def _update_tracks(self, detections, frame):
        """Enhanced tracking with occlusion handling"""
        active_ids = set()
        
        # Update existing tracks
        for det in detections:
            best_match = None
            min_distance = float('inf')
            
            for track_id, track in self.tracked_objects.items():
                # Calculate improved similarity score
                distance = self._get_similarity(track, det)
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                self._update_existing_track(best_match, det, frame)
                active_ids.add(best_match)
            else:
                new_id = self._create_new_track(det)
                active_ids.add(new_id)
        
        # Handle disappeared tracks
        self._cleanup_tracks(active_ids)

    def _get_similarity(self, track, detection):
        """Combined spatial and appearance similarity"""
        # Spatial distance
        track_center = track['center']
        det_center = ((detection['box'][0] + detection['box'][2]) // 2, 
                      (detection['box'][1] + detection['box'][3]) // 2)
        spatial_dist = np.sqrt((track_center[0]-det_center[0])**2 + 
                       (track_center[1]-det_center[1])**2)
        
        # Size similarity
        track_size = (track['box'][2] - track['box'][0]) * \
                     (track['box'][3] - track['box'][1])
        det_size = (detection['box'][2] - detection['box'][0]) * \
                   (detection['box'][3] - detection['box'][1])
        size_sim = abs(track_size - det_size) / max(track_size, det_size)
        
        return 0.7 * spatial_dist + 0.3 * size_sim * 100

    def _update_existing_track(self, track_id, detection, frame):
        """Update track with new detection"""
        track = self.tracked_objects[track_id]
        track['box'] = detection['box']
        track['center'] = ((detection['box'][0] + detection['box'][2]) // 2,
                          (detection['box'][1] + detection['box'][3]) // 2)
        track['last_seen'] = time.time()
        track['disappeared'] = 0
        
        # Emotion detection
        face_roi = self._get_valid_face_roi(frame, detection['box'])
        if face_roi is not None:
            emotion = self._detect_emotion(face_roi)
            self.emotion_history[track_id].append(emotion)

    def _create_new_track(self, detection):
        """Initialize new track"""
        new_id = self.next_id
        self.next_id += 1
        self.visitor_count += 1
        
        self.tracked_objects[new_id] = {
            'box': detection['box'],
            'center': ((detection['box'][0] + detection['box'][2]) // 2,
                      (detection['box'][1] + detection['box'][3]) // 2),
            'last_seen': time.time(),
            'disappeared': 0,
            'first_seen': time.time()
        }
        return new_id

    def _cleanup_tracks(self, active_ids):
        """Remove stale tracks"""
        current_time = time.time()
        to_delete = []
        
        for track_id, track in self.tracked_objects.items():
            if track_id not in active_ids:
                track['disappeared'] += 1
                if track['disappeared'] > self.max_disappeared:
                    to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracked_objects[track_id]

    def _get_valid_face_roi(self, frame, box):
        """Safe face extraction with boundary checks"""
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face = frame[y1:y2, x1:x2]
        return face if face.size > 0 else None

    def _detect_emotion(self, face_roi):
        """Robust emotion detection"""
        try:
            input_shape = self.fer_model.input(0).shape
            resized = cv2.resize(face_roi, (input_shape[3], input_shape[2]))
            input_data = np.transpose(resized, (2, 0, 1))  # HWC to CHW
            input_data = np.expand_dims(input_data, 0).astype(np.float32)
            input_data = (input_data - 127.5) / 127.5  # Normalization
            
            results = self.fer_model([input_data])[self.fer_model.output(0)]
            return self.emotion_labels[np.argmax(results)]
        except Exception as e:
            logger.warning(f"Emotion detection failed: {str(e)}")
            return "unknown"

    def _draw_analytics(self, frame):
        """Enhanced visualization"""
        # Draw tracks
        for track_id, track in self.tracked_objects.items():
            x1, y1, x2, y2 = track['box']
            
            # Get dominant emotion
            emotions = list(self.emotion_history[track_id])
            dominant = max(set(emotions), key=emotions.count) if emotions else "unknown"
            
            # Draw elements
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 
                       f"ID:{track_id} {dominant}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0,255,255), 1)
            
            # Draw path
            path = track.get('path', [])
            for i in range(1, len(path)):
                cv2.line(frame, path[i-1], path[i], (0,255,255), 1)
        
        # Draw counters
        cv2.putText(frame, 
                   f"Total: {self.visitor_count} | Active: {len(self.tracked_objects)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,255,255), 2)
        
        return frame

    def get_analytics(self):
        """Structured analytics data"""
        emotion_stats = defaultdict(int)
        for emotions in self.emotion_history.values():
            if emotions:
                dominant = max(set(emotions), key=emotions.count)
                emotion_stats[dominant] += 1
                
        return {
            'total_visitors': self.visitor_count,
            'active_visitors': len(self.tracked_objects),
            'emotion_stats': dict(emotion_stats),
            'timestamp': time.time()
        }
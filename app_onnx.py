from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os
from threading import Lock
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# Global variables
onnx_session = None
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

camera = None
is_camera_active = False
camera_lock = Lock()

def initialize_onnx_model(model_path='/Users/hares/Desktop/infor2025/project/intern/yolo/yolov8m.onnx'):
    """Initialize ONNX model"""
    global onnx_session
    
    if not os.path.exists(model_path):
        raise Exception(f"ONNX model not found at {model_path}")
    
    print(f"Loading ONNX model from {model_path}...")
    onnx_session = ort.InferenceSession(model_path)
    print("YOLOv8m ONNX model loaded successfully!")
    
    # Print model info
    input_info = onnx_session.get_inputs()[0]
    print(f"Model input shape: {input_info.shape}")
    print(f"Model input type: {input_info.type}")

def preprocess_image(image, input_size=640):
    """Preprocess image for ONNX inference"""
    original_height, original_width = image.shape[:2]
    
    # Calculate scale and padding
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create padded image
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    
    # Convert to RGB and normalize
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0
    
    # Transpose to CHW format and add batch dimension
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, pad_x, pad_y

def postprocess_onnx_output(outputs, scale, pad_x, pad_y, original_shape, conf_threshold=0.25, iou_threshold=0.45):
    """Postprocess ONNX model outputs"""
    try:
        # Handle different output formats
        if len(outputs) > 0:
            predictions = outputs[0]
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension if present
        else:
            return []
        
        print(f"Predictions shape: {predictions.shape}")
        
        # For YOLOv8, the output format is [num_detections, 84] (4 bbox + 80 classes)
        if predictions.shape[1] == 84:
            # No objectness score, just bbox + classes
            boxes = predictions[:, :4]  # x_center, y_center, w, h
            class_probs = predictions[:, 4:]  # class probabilities
            
            # Get class scores and IDs
            class_scores = np.max(class_probs, axis=1)
            class_ids = np.argmax(class_probs, axis=1)
            confidences = class_scores
            
        elif predictions.shape[1] == 85:
            # With objectness score
            boxes = predictions[:, :4]  # x_center, y_center, w, h
            scores = predictions[:, 4]  # objectness
            class_probs = predictions[:, 5:]  # class probabilities
            
            # Get class scores and IDs
            class_scores = np.max(class_probs, axis=1)
            class_ids = np.argmax(class_probs, axis=1)
            confidences = scores * class_scores
        else:
            print(f"Unexpected output shape: {predictions.shape}")
            return []
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Filter by confidence
        valid_indices = confidences > conf_threshold
        
        if not np.any(valid_indices):
            return []
        
        x1 = x1[valid_indices]
        y1 = y1[valid_indices]
        x2 = x2[valid_indices]
        y2 = y2[valid_indices]
        confidences = confidences[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # Adjust coordinates for padding and scaling
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        # Prepare for NMS
        boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h format for OpenCV
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(), 
            confidences.tolist(), 
            conf_threshold, 
            iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Format results
        results = []
        for i in indices.flatten():
            if i < len(x1):  # Safety check
                results.append({
                    'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': class_names[class_ids[i]] if class_ids[i] < len(class_names) else 'unknown'
                })
        
        return results
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return []

def run_onnx_inference(frame):
    """Run inference using ONNX model"""
    global onnx_session
    
    if onnx_session is None:
        raise Exception("ONNX model not loaded")
    
    # Preprocess
    input_tensor, scale, pad_x, pad_y = preprocess_image(frame)
    
    # Run inference
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: input_tensor})
    
    # Postprocess
    detections = postprocess_onnx_output(outputs, scale, pad_x, pad_y, frame.shape)
    
    return detections

def draw_detections(frame, detections):
    """Draw detection boxes and labels on frame"""
    annotated_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(x) for x in det['bbox']]
        confidence = det['confidence']
        class_name = det['class_name']
        
        # Choose color based on class
        color = (0, 255, 0)  # Green default
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return annotated_frame

def initialize_camera():
    """Initialize camera"""
    global camera, is_camera_active
    with camera_lock:
        if camera is None:
            print("Initializing camera...")
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                print("Error: Could not open camera")
                # Try different camera indices
                for i in range(1, 5):
                    print(f"Trying camera index {i}")
                    camera = cv2.VideoCapture(i)
                    if camera.isOpened():
                        print(f"Camera opened successfully with index {i}")
                        break
                else:
                    camera = None
                    raise Exception("Could not open any camera")
            
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            print("Camera initialized successfully!")
        is_camera_active = True

def release_camera():
    """Release camera resources"""
    global camera, is_camera_active
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        is_camera_active = False

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera detection"""
    try:
        if onnx_session is None:
            initialize_onnx_model()
        initialize_camera()
        return jsonify({'status': 'success', 'message': 'Camera started with ONNX model'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera detection"""
    try:
        release_camera()
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera_feed')
def camera_feed():
    """Stream camera feed with YOLO detection"""
    def generate_frames():
        global camera, is_camera_active, onnx_session
        frame_count = 0
        
        print("Camera feed requested, is_camera_active:", is_camera_active)
        
        while is_camera_active and camera is not None:
            try:
                with camera_lock:
                    if camera is None:
                        print("Camera is None, breaking")
                        break
                        
                    ret, frame = camera.read()
                    
                    if not ret:
                        print("Failed to read frame from camera")
                        break
                    
                    print(f"Frame {frame_count} captured, shape: {frame.shape}")
                    
                    # Try to run ONNX inference, fallback to simple frame if it fails
                    try:
                        detections = run_onnx_inference(frame)
                        print(f"ONNX inference successful, {len(detections)} detections")
                        annotated_frame = draw_detections(frame, detections)
                    except Exception as e:
                        print(f"ONNX inference failed: {e}")
                        # Use original frame if inference fails
                        annotated_frame = frame.copy()
                        cv2.putText(annotated_frame, f"Inference Error: {str(e)[:50]}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Add frame info
                    frame_count += 1
                    cv2.putText(annotated_frame, f"Frame: {frame_count} (ONNX)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert frame to JPEG
                    ret_encode, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret_encode:
                        print("Failed to encode frame")
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            except Exception as e:
                print(f"Error in camera feed generation: {e}")
                # Try to send an error frame
                try:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Error: {str(e)}", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except:
                    pass
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detect_image', methods=['POST'])
def detect_image():
    """Detect objects in uploaded image"""
    try:
        if onnx_session is None:
            initialize_onnx_model()
            
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No image selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run ONNX inference
        detections = run_onnx_inference(frame)
        
        # Draw results on frame
        annotated_frame = draw_detections(frame, detections)
        
        # Convert to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format detections for response
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                'class': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox']
            })
        
        return jsonify({
            'status': 'success',
            'detections': formatted_detections,
            'annotated_image': img_base64
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detect_video', methods=['POST'])
def detect_video():
    """Process uploaded video with ROI parking spots"""
    try:
        if onnx_session is None:
            initialize_onnx_model()
            
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No video selected'}), 400
        
        # Get parking spots from request
        parking_spots = []
        if 'parking_spots' in request.form:
            parking_spots = json.loads(request.form['parking_spots'])
        
        # Save uploaded video temporarily
        video_path = f"temp_video_{file.filename}"
        file.save(video_path)
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        results_data = []
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run ONNX detection
            detections = run_onnx_inference(frame)
            
            # Check each parking spot
            spot_status = []
            for spot in parking_spots:
                spot_id = spot['id']
                points = np.array(spot['points'], np.int32)
                
                # Create mask for ROI
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                
                # Check if any vehicle detection overlaps with this spot
                occupied = False
                for det in detections:
                    class_name = det['class_name']
                    if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        # Get bounding box center
                        x1, y1, x2, y2 = det['bbox']
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Check if center is in parking spot
                        if center_y < mask.shape[0] and center_x < mask.shape[1]:
                            if mask[center_y, center_x] > 0:
                                occupied = True
                                break
                
                spot_status.append({
                    'id': spot_id,
                    'occupied': occupied
                })
            
            results_data.append({
                'frame': frame_number,
                'timestamp': frame_number / cap.get(cv2.CAP_PROP_FPS),
                'spots': spot_status
            })
            
            frame_number += 1
        
        cap.release()
        
        # Clean up temporary file
        os.remove(video_path)
        
        return jsonify({
            'status': 'success',
            'results': results_data,
            'total_frames': frame_number
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'model_loaded': onnx_session is not None,
        'camera_active': is_camera_active,
        'model_type': 'ONNX'
    })

@app.route('/api/camera_test')
def camera_test():
    """Test camera feed without ONNX inference"""
    def generate_frames():
        global camera, is_camera_active
        frame_count = 0
        
        print("Camera test feed requested")
        
        while is_camera_active and camera is not None:
            try:
                with camera_lock:
                    if camera is None:
                        break
                        
                    ret, frame = camera.read()
                    
                    if not ret:
                        print("Failed to read frame from camera")
                        break
                    
                    # Add simple frame counter (no ONNX inference)
                    frame_count += 1
                    cv2.putText(frame, f"Test Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert frame to JPEG
                    ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret_encode:
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            except Exception as e:
                print(f"Error in camera test: {e}")
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize ONNX model on startup
    try:
        initialize_onnx_model()
    except Exception as e:
        print(f"Warning: Could not load ONNX model on startup: {e}")
        print("Model will be loaded when first needed")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)

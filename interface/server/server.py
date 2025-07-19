from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import uvicorn
import threading
import time
import base64
import tempfile
import os
import json

# --- Path setup ---
# Get the absolute path of the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path: str) -> str:
    """Construct an absolute path from a path relative to the script's directory."""
    return os.path.join(BASE_DIR, relative_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and parking configuration on startup and cleanup on shutdown"""
    # Declare global variables that will be used
    global model, detection_running, cap, uploaded_video_path
    try:
        print("🚀 Starting up Parking Detection API...")
        initialize_model()
        load_parking_configuration()
        print("✅ Startup completed successfully")
        yield
    except Exception as e:
        print(f"❌ Startup error: {e}")
        # If startup fails, ensure to yield so the app can at least try to start,
        # but the /start-detection endpoint will likely fail without a model.
        yield
    finally:
        # Cleanup on shutdown
        print("🛑 Shutting down Parking Detection API...")
        try:
            if detection_running:
                detection_running = False
            if cap:
                cap.release()
            if uploaded_video_path and os.path.exists(uploaded_video_path):
                os.unlink(uploaded_video_path)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

# Initialize FastAPI app
app = FastAPI(title="Parking Lot Detection API", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DetectionResult(BaseModel):
    objects: List[Dict[str, Any]]
    count: int
    timestamp: str

class ParkingStatus(BaseModel):
    total_spots: int
    occupied_spots: int
    available_spots: int
    occupancy_status: List[bool]
    timestamp: str

class VideoConfig(BaseModel):
    video_file: Optional[str] = None # Path to the video file
    confidence: float = 0.5
    draw_car_bounding_box: Optional[bool] = None # Will be set by config.json or default
    draw_dot_on_car: Optional[bool] = None # Will be set by config.json or default

# Global variables
model = None
detection_running = False
current_detection_result = None
current_parking_status = None
detection_thread = None
cap = None
latest_frame = None
frame_lock = threading.Lock()
current_config = None
uploaded_video_path = None # For dynamically uploaded video files
parking_areas = []
draw_car_bounding_box = True # Default from main.py
draw_dot_on_car = True       # Default from main.py

def initialize_model():
    """Initialize YOLO model"""
    global model
    try:
        print("🔄 Initializing YOLO model...")
        model_path_custom = get_path('model_with_83_image.pt')
        model_path_general = get_path('yolov8s.pt')

        if os.path.exists(model_path_custom):
            model = YOLO(model_path_custom)
            print(f"✅ Custom parking model '{model_path_custom}' loaded")
        elif os.path.exists(model_path_general):
            model = YOLO(model_path_general)
            print(f"✅ General YOLO model '{model_path_general}' loaded from local file")
        else:
            # This will download yolov8s.pt to the script's directory if not found
            model = YOLO('yolov8s.pt')
            print("✅ General YOLO model 'yolov8s.pt' downloaded and loaded")
    except Exception as e:
        print(f"❌ Error initializing YOLO model: {e}")
        print("⚠️ Continuing without model - detection will be disabled")
        model = None

def load_parking_configuration():
    """Load parking spots and configuration"""
    global parking_areas, draw_car_bounding_box, draw_dot_on_car
    
    # Load parking spots
    parking_spots_path = get_path('parking_spots.json')
    try:
        with open(parking_spots_path, 'r') as f:
            parking_slots_json = json.load(f)
            parking_areas = []
            for slot in parking_slots_json:
                points = np.array(slot['points'], np.int32)
                parking_areas.append(points)
        print(f"✅ Loaded {len(parking_areas)} parking spots from {parking_spots_path}")
    except FileNotFoundError:
        print(f"⚠️ '{parking_spots_path}' file not found. Running without predefined parking spots.")
        parking_areas = [] # Ensure parking_areas is empty if file not found
    except Exception as e:
        print(f"❌ Error loading parking spots: {e}")
        parking_areas = [] # Ensure parking_areas is empty on other errors

def parking_detection_loop():
    """Parking lot detection loop with YOLO object detection"""
    global detection_running, current_detection_result, current_parking_status, cap, model, latest_frame, frame_lock, current_config
    global parking_areas, draw_car_bounding_box, draw_dot_on_car # Use global drawing flags
    
    if model is None:
        print("❌ YOLO model not initialized. Stopping detection loop.")
        detection_running = False # Ensure loop stops if model is not ready
        return
    
    # Determine video source based on current_config
    video_source_path = None
    if current_config and current_config.video_file:
        video_source_path = current_config.video_file
    elif uploaded_video_path: # Fallback to uploaded video
        video_source_path = uploaded_video_path
    else:
        print("❌ No video file specified.")
        detection_running = False
        return
        
    print(f"🎥 Using video file: {video_source_path}")
    cap = cv2.VideoCapture(video_source_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_source_path}")
        detection_running = False
        return
    
    print("✅ Video source opened successfully")
    print("🔄 Starting parking detection loop...")
    
    # Initialize latest_frame with a placeholder
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Parking Detection Starting...", (120, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    with frame_lock:
        latest_frame = placeholder
    
    # Use confidence from config, otherwise default
    confidence_threshold = current_config.confidence if current_config else 0.5
    
    # Use drawing flags from loaded config.json, which are global
    # (current_config.draw_car_bounding_box and current_config.draw_dot_on_car
    # are Optional, so they don't override the global settings if not explicitly set in the config object from the request body)
    
    while detection_running:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from video source")
            # For video files, restart if reached end
            if current_config and current_config.video_file and video_source_path:
                print("📹 Video file ended, restarting from beginning")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from beginning
                continue
            # If we can't read frames, stop detection
            print("❌ Cannot read frames. Stopping detection.")
            detection_running = False
            break
        
        try:
            # Initialize occupancy status for parking spots
            occupancy_status = [False] * len(parking_areas) if parking_areas else []
            
            # Debug: Print parking areas info periodically (every 30 frames)
            if len(parking_areas) > 0 and hasattr(parking_detection_loop, 'frame_count'):
                parking_detection_loop.frame_count += 1
                if parking_detection_loop.frame_count % 30 == 0:  # Every 30 frames (about 1 second at 30 FPS)
                    print(f"🅿️ Debug: {len(parking_areas)} parking areas configured")
            elif not hasattr(parking_detection_loop, 'frame_count'):
                parking_detection_loop.frame_count = 0
                if len(parking_areas) > 0:
                    print(f"🅿️ Starting detection with {len(parking_areas)} parking areas configured")
                else:
                    print("⚠️ No parking areas configured - occupancy detection disabled")
            
            # Run YOLO detection
            results = model(frame, conf=confidence_threshold)
            
            # Create a copy of the frame for drawing, so the original can be used for detection
            display_frame = frame.copy()
            
            # Process each detected object and draw on display_frame
            detected_objects = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class name and check if it's a vehicle
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Calculate center point for parking spot detection
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        point_to_test = (float(center_x), float(center_y))
                        
                        detected_objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y]
                        })
                        
                        # Debug: Print detected object classes to understand what's being detected
                        if hasattr(parking_detection_loop, 'class_debug_count'):
                            parking_detection_loop.class_debug_count += 1
                            if parking_detection_loop.class_debug_count % 30 == 0:  # Every 30 detections
                                print(f"🔍 Detected: {class_name} (confidence: {confidence:.2f}) at ({center_x}, {center_y})")
                        else:
                            parking_detection_loop.class_debug_count = 0
                            print(f"🔍 First detection: {class_name} (confidence: {confidence:.2f}) at ({center_x}, {center_y})")
                        
                        # Conditionally draw the dot in the center of the car
                        if draw_dot_on_car:
                            dot_color = (42, 42, 165)  # Brown color in BGR
                            cv2.circle(display_frame, (center_x, center_y), 5, dot_color, -1)
                        
                        # Check which parking spots this vehicle occupies
                        # Expanded list of vehicle classes that could occupy parking spots
                        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle', 'auto', 'SUV', 'sedan', 'van']
                        
                        # Check for parking spot occupancy for ALL detected objects, not just vehicles
                        # This is more flexible and catches any object that might be in a parking spot
                        for i, area in enumerate(parking_areas):
                            if cv2.pointPolygonTest(area, point_to_test, False) >= 0:
                                occupancy_status[i] = True
                                if hasattr(parking_detection_loop, 'spot_debug_count'):
                                    parking_detection_loop.spot_debug_count += 1
                                    if parking_detection_loop.spot_debug_count % 30 == 0:  # Every 30 spot detections
                                        print(f"🅿️ Object '{class_name}' detected in parking spot {i+1} at ({center_x}, {center_y})")
                                else:
                                    parking_detection_loop.spot_debug_count = 0
                                    print(f"🅿️ First spot detection: '{class_name}' in parking spot {i+1} at ({center_x}, {center_y})")
                                break # An object can only occupy one spot

                        # Conditionally draw the bounding box
                        if draw_car_bounding_box:
                            box_color = (255, 255, 255)  # White
                            box_thickness = 2
                            label = f"{class_name} {confidence:.2f}"
                            font_color = (0, 0, 0)  # Black
                            font_scale = 0.5
                            font_thickness = 1

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(display_frame, (x1, y1 - text_height - 10),
                                          (x1 + text_width, y1), box_color, -1)
                            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, font_color, font_thickness)
            
            # Draw parking spots with occupancy status on display_frame
            for i, area in enumerate(parking_areas):
                if occupancy_status[i]:
                    color = (0, 0, 255)  # Red: Occupied
                else:
                    color = (0, 255, 0)  # Green: Empty
                cv2.polylines(display_frame, [area], isClosed=True,
                              color=color, thickness=3)

            # Update detection results
            current_detection_result = {
                'objects': detected_objects,
                'count': len(detected_objects),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update parking status
            if parking_areas:
                occupied_count = sum(occupancy_status)
                total_spots = len(parking_areas)
                available_spots = total_spots - occupied_count
                
                current_parking_status = {
                    'total_spots': total_spots,
                    'occupied_spots': occupied_count,
                    'available_spots': available_spots,
                    'occupancy_status': occupancy_status,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"🅿️ Parking: {occupied_count}/{total_spots} spots occupied, {available_spots} available")
            else:
                print(f"🚗 Detected {len(detected_objects)} vehicles (no parking spots defined)")
            
            # Store the annotated display_frame for streaming
            with frame_lock:
                latest_frame = display_frame
            
            # Sleep to prevent excessive CPU usage
            time.sleep(0.033) # Roughly 30 FPS, adjust as needed

        except Exception as e:
            print(f"❌ Detection error: {e}")
            # Do not break immediately, try to continue, but log the error
            time.sleep(1) # Wait a bit before trying the next frame
    
    if cap:
        cap.release()
    print("🛑 Parking detection loop stopped")

# API Endpoints

@app.get("/api/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Parking Lot Detection API is running", "status": "healthy"}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    global uploaded_video_path
    
    # Check if file is a video
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create the 'videos' directory if it doesn't exist
    videos_dir = get_path("videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Save the uploaded file with its original filename
    file_path = os.path.join(videos_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    uploaded_video_path = file_path
    
    print(f"📹 Video uploaded to: {file_path}")
    return {"message": "Video uploaded successfully", "filename": file.filename, "filepath": file_path}

@app.post("/api/start-detection")
async def start_detection(config: Optional[VideoConfig] = None):
    """Start YOLO detection with a hardcoded video file"""
    global detection_running, detection_thread, current_config, model
    
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model not available. Please check server logs.")
    
    if detection_running:
        return {"message": "Detection already running", "status": "running"}
    
    # Store configuration, but override the video file path
    current_config = config or VideoConfig()
    
    # Hardcode the video path
    video_path = "/Users/hares/Desktop/infor2025/project/intern/interface/server/video/vid01.mp4"
    
    if not os.path.exists(video_path):
        # If the primary hardcoded video is not found, try a fallback in the local 'videos' folder
        fallback_path = get_path('video/vid01.mp4')
        if os.path.exists(fallback_path):
            video_path = fallback_path
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found at hardcoded paths: {video_path} or {fallback_path}")
            
    current_config.video_file = video_path

    try:
        detection_running = True
        
        # Start detection in background thread
        detection_thread = threading.Thread(target=parking_detection_loop)
        detection_thread.daemon = True # Allow main program to exit even if thread is running
        detection_thread.start()
        
        source_msg = f"video file ({current_config.video_file})"
        
        print(f"🚀 Detection started with {source_msg}")
        return {
            "message": f"Parking detection started successfully with {source_msg}",
            "status": "started"
        }
    except Exception as e:
        detection_running = False
        raise HTTPException(status_code=500, detail=f"Failed to start detection: {str(e)}")

@app.post("/api/stop-detection")
async def stop_detection():
    """Stop detection"""
    global detection_running, detection_thread, current_detection_result, current_parking_status, cap, latest_frame, frame_lock, uploaded_video_path
    
    if not detection_running:
        return {"message": "Detection not running", "status": "stopped"}
    
    detection_running = False
    current_detection_result = None
    current_parking_status = None
    
    # Clear the latest frame
    with frame_lock:
        latest_frame = None
    
    # Wait for thread to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=5) # Give it 5 seconds to finish cleanly
        if detection_thread.is_alive():
            print("⚠️ Detection thread did not terminate in time.")
    
    if cap:
        cap.release()
        cap = None
    
    # We are no longer using temporary files, so we won't clean up the video on stop.
    # The user can manage the videos in the 'videos' folder.
    if uploaded_video_path:
        uploaded_video_path = None # Reset the path
    
    print("🛑 Parking detection stopped")
    return {"message": "Parking detection stopped successfully", "status": "stopped"}

@app.get("/api/detection-status")
async def get_detection_status():
    """Get current detection status and results"""
    global detection_running, current_detection_result, current_parking_status
    
    return {
        "is_running": detection_running,
        "detection_result": current_detection_result,
        "parking_status": current_parking_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/detection-result")
async def get_detection_result():
    """Get detection result"""
    global current_detection_result
    
    if current_detection_result is None:
        raise HTTPException(status_code=404, detail="No detection result available")
    
    return current_detection_result

@app.get("/api/parking-status")
async def get_parking_status():
    """Get current parking lot status"""
    global current_parking_status
    
    if current_parking_status is None:
        # Return a default empty status if no detection has run yet
        return {
            "message": "No parking status available. Start detection.",
            "total_spots": len(parking_areas), # Still show total spots from loaded config
            "occupied_spots": 0,
            "available_spots": len(parking_areas),
            "occupancy_status": [False] * len(parking_areas),
            "timestamp": datetime.now().isoformat()
        }
    
    return current_parking_status

@app.post("/api/upload-parking-config")
async def upload_parking_config(file: UploadFile = File(...)):
    """Upload parking spots configuration JSON file"""
    
    # Check if file is JSON
    if not file.content_type == 'application/json':
        raise HTTPException(status_code=400, detail="File must be JSON")
    
    parking_spots_path = get_path('parking_spots.json')
    try:
        content = await file.read()
        parking_data = json.loads(content.decode('utf-8'))
        
        # Save to parking_spots.json
        with open(parking_spots_path, 'w') as f:
            json.dump(parking_data, f, indent=2)
        
        # Reload parking configuration including updated parking_areas
        load_parking_configuration()
        
        return {
            "message": "Parking configuration uploaded successfully", 
            "spots_loaded": len(parking_areas),
            "filename": file.filename
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in parking spots file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing parking config file: {str(e)}")

@app.post("/api/update-drawing-config")
async def update_drawing_config(config_data: Dict[str, bool]):
    """
    Update drawing configuration (draw_car_box, draw_dot_car) and save to config.json.
    Expects a JSON body like: {"draw_car_box": true, "draw_dot_car": false}
    """
    global draw_car_bounding_box, draw_dot_on_car

    config_path = get_path('config.json')
    try:
        # Load existing config or initialize an empty dict
        current_app_config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    current_app_config = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Existing config.json is invalid. Creating new config.")

        # Update values from the request body
        if 'draw_car_box' in config_data:
            draw_car_bounding_box = bool(config_data['draw_car_box'])
            current_app_config['draw_car_box'] = draw_car_bounding_box
        
        if 'draw_dot_car' in config_data:
            draw_dot_on_car = bool(config_data['draw_dot_car'])
            current_app_config['draw_dot_car'] = draw_dot_on_car

        # Save updated config back to file
        with open(config_path, 'w') as f:
            json.dump(current_app_config, f, indent=2)
        
        print(f"✅ Drawing configuration updated: draw_car_box={draw_car_bounding_box}, draw_dot_car={draw_dot_on_car}")

        return {
            "message": "Drawing configuration updated successfully",
            "draw_car_bounding_box": draw_car_bounding_box,
            "draw_dot_on_car": draw_dot_on_car
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating drawing configuration: {str(e)}")


@app.get("/api/parking-config")
async def get_parking_config():
    """Get current parking configuration and drawing settings"""
    return {
        "total_parking_spots": len(parking_areas),
        "draw_car_bounding_box": draw_car_bounding_box,
        "draw_dot_on_car": draw_dot_on_car,
        "has_parking_spots": len(parking_areas) > 0
    }

@app.get("/api/video-feed")


@app.get("/api/video-feed")
async def video_feed():
    """Stream single frame of video feed with parking detection annotations as Base64"""
    global latest_frame, frame_lock, detection_running
    
    if not detection_running:
        # If detection is not running, return a placeholder indicating so
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Detection Not Running", (120, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red text
        _, buffer = cv2.imencode('.jpg', placeholder)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": f"data:image/jpeg;base64,{img_base64}", "timestamp": datetime.now().isoformat()}

    if latest_frame is None:
        # Return a placeholder frame if detection is running but no frame is available yet
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing parking detection...", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', placeholder)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": f"data:image/jpeg;base64,{img_base64}", "timestamp": datetime.now().isoformat()}
    
    with frame_lock:
        # The frame already has annotations drawn in the detection loop
        display_frame = latest_frame.copy()
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', display_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"image": f"data:image/jpeg;base64,{img_base64}", "timestamp": datetime.now().isoformat()}

@app.post("/api/configure-roi")
async def configure_roi(roi_data: Dict[str, Any]):
    """Configure ROI (Region of Interest) areas for parking detection"""
    global parking_areas
    
    try:
        if 'regions' in roi_data and roi_data['regions']:
            # Convert client ROI data to server format
            parking_areas = []
            
            for region in roi_data['regions']:
                # Extract points from the region data
                if 'points' in region:
                    # Convert points to numpy array format expected by cv2.pointPolygonTest
                    points = np.array(region['points'], np.int32)
                    parking_areas.append(points)
            
            print(f"✅ ROI Configuration updated: {len(parking_areas)} parking regions configured")
            
            return {
                "message": "ROI configuration updated successfully",
                "regions_configured": len(parking_areas),
                "status": "configured"
            }
        else:
            # Clear parking areas if no regions provided
            parking_areas = []
            print("🔄 ROI Configuration cleared: No parking regions")
            
            return {
                "message": "ROI configuration cleared",
                "regions_configured": 0,
                "status": "cleared"
            }
            
    except Exception as e:
        print(f"❌ Error configuring ROI: {e}")
        raise HTTPException(status_code=500, detail=f"Error configuring ROI: {str(e)}")

@app.get("/api/video-stream")
async def video_stream():
    """Stream video feed as MJPEG with parking detection annotations"""
    def generate():
        while detection_running:
            if latest_frame is not None:
                with frame_lock:
                    # The frame already has annotations drawn in the detection loop
                    display_frame = latest_frame.copy()
                    
                    _, buffer = cv2.imencode('.jpg', display_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # If no frame is available, send a placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No video feed available", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # Roughly 30 FPS, adjust as needed
    
    # Check if detection is running, if not, return an error or a static image
    if not detection_running:
        raise HTTPException(status_code=400, detail="Video detection is not running. Please start detection first.")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

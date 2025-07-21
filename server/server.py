from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import uvicorn
import threading
import time
import os
import json
import asyncio

# --- Path setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path: str) -> str:
    """Construct an absolute path from a path relative to the script's directory."""
    return os.path.join(BASE_DIR, relative_path)

# Initialize FastAPI app
app = FastAPI(title="Parking Lot Detection API", version="1.0.0")

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VideoConfig(BaseModel):
    confidence: float = 0.5

# Global variables
model = None
detection_running = False
detection_thread = None
cap = None
latest_frame = None
frame_lock = threading.Lock()
parking_areas = []
# Hardcoded video path - using vid01.mp4 as default
HARDCODED_VIDEO_PATH = get_path('video/vid01.mp4')

def initialize_model():
    """Initialize YOLO model"""
    global model
    try:
        print("🔄 Initializing YOLO model...")
        model_path = get_path('model_with_83_image.pt')
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ YOLO model loaded from {model_path}")
        else:
            model = YOLO('yolov8s.pt')  # This will download if needed
            print("✅ YOLO model downloaded and loaded")
    except Exception as e:
        print(f"❌ Error initializing YOLO model: {e}")
        model = None

def load_parking_spots():
    """Load parking spots from JSON file"""
    global parking_areas
    
    parking_spots_path = get_path('parking_spots.json')
    try:
        with open(parking_spots_path, 'r') as f:
            parking_slots_json = json.load(f)
            parking_areas = []
            for slot in parking_slots_json:
                points = np.array(slot['points'], np.int32)
                parking_areas.append(points)
        print(f"✅ Loaded {len(parking_areas)} parking spots")
    except FileNotFoundError:
        print("⚠️ No parking spots file found")
        parking_areas = []
    except Exception as e:
        print(f"❌ Error loading parking spots: {e}")
        parking_areas = []

def detection_loop():
    """Main detection loop"""
    global detection_running, cap, model, latest_frame, frame_lock, parking_areas
    
    if model is None:
        print("❌ Model not loaded")
        detection_running = False
        return
    
    # Use hardcoded video path
    video_path = HARDCODED_VIDEO_PATH
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        detection_running = False
        return
        
    print(f"🎥 Using hardcoded video: {video_path}")
    
    # Initialize video capture with error handling
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            detection_running = False
            return
    except Exception as e:
        print(f"❌ Error opening video capture: {e}")
        detection_running = False
        return
    
    print("🔄 Starting detection...")
    
    try:
        while detection_running:
            ret, frame = cap.read()
            if not ret:
                # Restart video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Check if we should stop before processing
            if not detection_running:
                break
            
            try:
                # Run YOLO detection
                results = model(frame, conf=0.5)
                
                # Create display frame
                display_frame = frame.copy()
                
                # Track parking spot occupancy
                occupancy_status = [False] * len(parking_areas)
                
                # Process detections
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get detection info
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Calculate center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            
                            # Draw label
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(display_frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Check parking spot occupancy
                            point_to_test = (float(center_x), float(center_y))
                            for i, area in enumerate(parking_areas):
                                if cv2.pointPolygonTest(area, point_to_test, False) >= 0:
                                    occupancy_status[i] = True
                                    break
                
                # Draw parking spots
                for i, area in enumerate(parking_areas):
                    color = (0, 0, 255) if occupancy_status[i] else (0, 255, 0)  # Red if occupied, Green if free
                    cv2.polylines(display_frame, [area], isClosed=True, color=color, thickness=3)
                    
                    # Add spot number
                    center = area.mean(axis=0).astype(int)
                    cv2.putText(display_frame, str(i+1), tuple(center), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Update latest frame
                with frame_lock:
                    latest_frame = display_frame.copy() if detection_running else None
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                if detection_running:  # Only sleep if still running
                    time.sleep(1)
    
    except Exception as e:
        print(f"❌ Critical error in detection loop: {e}")
    
    finally:
        # Cleanup resources
        try:
            if cap is not None:
                cap.release()
                print("� Video capture released in detection loop")
        except Exception as e:
            print(f"⚠️ Error releasing video capture in detection loop: {e}")
        
        # Clear latest frame
        with frame_lock:
            latest_frame = None
            
        print("🛑 Detection loop exited")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_model()
    load_parking_spots()

# API Endpoints

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Parking Detection Server is running", "status": "connected"}

@app.post("/start-detection")
async def start_detection():
    """Start detection"""
    global detection_running, detection_thread, model, cap, latest_frame
    
    if model is None:
        print("❌ Model not loaded, attempting to reinitialize...")
        initialize_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded and cannot be initialized")
    
    if detection_running:
        return {"message": "Detection already running"}
    
    # Ensure any previous resources are cleaned up
    if cap is not None:
        try:
            cap.release()
            cap = None
            print("🧹 Cleaned up previous video capture")
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")
    
    # Clear any previous frame
    with frame_lock:
        latest_frame = None
    
    # Wait for any previous thread to finish
    if detection_thread is not None and detection_thread.is_alive():
        print("⏳ Waiting for previous detection thread to finish...")
        detection_thread.join(timeout=5)
        if detection_thread.is_alive():
            print("⚠️ Warning: Previous thread did not stop, starting anyway")
    
    detection_running = True
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()
    
    print("🚀 Detection started")
    return {"message": "Detection started successfully"}

@app.post("/stop-detection")
async def stop_detection():
    """Stop detection"""
    global detection_running, detection_thread, cap, latest_frame, frame_lock
    
    if not detection_running:
        return {"message": "Detection not running"}
    
    print("🛑 Stopping detection...")
    detection_running = False
    
    # Give the detection loop time to exit gracefully
    await asyncio.sleep(0.1)
    
    # Wait for thread to finish with longer timeout
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=10)
        if detection_thread.is_alive():
            print("⚠️ Warning: Detection thread did not stop gracefully")
    
    # Ensure video capture is properly released
    try:
        if cap is not None:
            cap.release()
            cap = None
            print("📹 Video capture released")
    except Exception as e:
        print(f"⚠️ Warning: Error releasing video capture: {e}")
    
    # Clear the latest frame
    with frame_lock:
        latest_frame = None
    
    # Reset thread reference
    detection_thread = None
    
    print("✅ Detection stopped successfully")
    return {"message": "Detection stopped successfully"}

@app.get("/status")
async def get_status():
    """Get system status"""
    global detection_running, model, parking_areas
    
    # Check if hardcoded video file exists
    video_exists = os.path.exists(HARDCODED_VIDEO_PATH)
    
    return {
        "backend_connected": True,
        "detection_status": "running" if detection_running else "stopped",
        "model_loaded": model is not None,
        "video_available": video_exists,
        "video_path": HARDCODED_VIDEO_PATH if video_exists else "Not found",
        "total_parking_spots": len(parking_areas),
        "has_parking_config": len(parking_areas) > 0,
        "parking_areas_loaded": len(parking_areas) > 0
    }

@app.get("/video-stream")
async def video_stream():
    """Stream video with detection annotations"""
    def generate():
        try:
            while detection_running:
                if latest_frame is not None:
                    try:
                        with frame_lock:
                            if latest_frame is not None:  # Double check after acquiring lock
                                frame = latest_frame.copy()
                                _, buffer = cv2.imencode('.jpg', frame)
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                            else:
                                # Frame became None while acquiring lock
                                break
                    except Exception as e:
                        print(f"⚠️ Error encoding frame: {e}")
                        break
                else:
                    # Placeholder frame
                    try:
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Initializing...", (200, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        _, buffer = cv2.imencode('.jpg', placeholder)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    except Exception as e:
                        print(f"⚠️ Error creating placeholder frame: {e}")
                        break
                
                time.sleep(0.033)
        except Exception as e:
            print(f"❌ Error in video stream generator: {e}")
    
    if not detection_running:
        raise HTTPException(status_code=400, detail="Detection not running. Start detection first.")
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload-roi")
async def upload_roi(file: UploadFile = File(...)):
    """Upload ROI configuration JSON file"""
    global parking_areas
    
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Read the uploaded JSON content
        content = await file.read()
        roi_data = json.loads(content.decode('utf-8'))
        
        # Validate and process the ROI data
        if isinstance(roi_data, list):
            # Direct list format
            parking_spots = roi_data
        elif isinstance(roi_data, dict) and 'parking_spots' in roi_data:
            # Object with parking_spots property
            parking_spots = roi_data['parking_spots']
        else:
            raise HTTPException(status_code=400, detail="Invalid JSON format. Expected array or object with 'parking_spots' property.")
        
        # Update parking areas
        parking_areas = []
        for spot in parking_spots:
            if 'points' in spot:
                points = np.array(spot['points'], np.int32)
                parking_areas.append(points)
        
        # Save to file
        parking_spots_path = get_path('parking_spots.json')
        with open(parking_spots_path, 'w') as f:
            json.dump(parking_spots, f, indent=2)
        
        print(f"✅ ROI uploaded and saved: {len(parking_areas)} parking spots")
        return {
            "message": "ROI configuration uploaded successfully", 
            "spots_configured": len(parking_areas),
            "filename": file.filename
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        print(f"❌ Error uploading ROI: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading ROI: {str(e)}")

@app.post("/update-roi")
async def update_roi(roi_config: Dict[str, Any]):
    """Update ROI configuration"""
    global parking_areas
    
    try:
        if 'parking_spots' in roi_config and roi_config['parking_spots']:
            # Convert to numpy arrays
            parking_areas = []
            for spot in roi_config['parking_spots']:
                if 'points' in spot:
                    points = np.array(spot['points'], np.int32)
                    parking_areas.append(points)
            
            # Save to file
            parking_spots_path = get_path('parking_spots.json')
            with open(parking_spots_path, 'w') as f:
                json.dump(roi_config['parking_spots'], f, indent=2)
            
            print(f"✅ ROI updated: {len(parking_areas)} parking spots")
            return {"message": "ROI updated successfully", "spots_configured": len(parking_areas)}
        else:
            # Clear parking areas
            parking_areas = []
            parking_spots_path = get_path('parking_spots.json')
            with open(parking_spots_path, 'w') as f:
                json.dump([], f, indent=2)
            
            print("🔄 ROI cleared")
            return {"message": "ROI cleared", "spots_configured": 0}
            
    except Exception as e:
        print(f"❌ Error updating ROI: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating ROI: {str(e)}")

@app.get("/get-roi")
async def get_roi():
    """Get current ROI configuration"""
    parking_spots_path = get_path('parking_spots.json')
    
    try:
        if os.path.exists(parking_spots_path):
            with open(parking_spots_path, 'r') as f:
                parking_spots = json.load(f)
            return {"parking_spots": parking_spots, "total_spots": len(parking_spots)}
        else:
            return {"parking_spots": [], "total_spots": 0}
    except Exception as e:
        print(f"❌ Error getting ROI: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting ROI: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Parking Detection Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

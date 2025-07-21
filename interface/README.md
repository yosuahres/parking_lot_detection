# Parking Lot Detection System

A web-based parking lot detection system with ROI configuration capabilities.

## Project Structure

```
client/
├── index.html          # Main page - displays video detection
└── roi-config.html     # ROI configuration page

server/
├── server.py          # FastAPI backend server
├── parking_spots.json # ROI configuration storage
├── model_with_83_image.pt # Pre-trained parking detection model
├── yolov8n.pt        # YOLO model
├── yolov8s.pt        # YOLO model
└── video/            # Video files directory
    ├── vid01.mkv
    ├── vid01.mp4
    ├── vid03.mkv
    └── vid04.mkv
```

## Features

### Main Page (index.html)
- Real-time video streaming with parking detection
- Upload video files for detection
- Start/stop detection controls
- View current detection status
- Navigate to ROI configuration

### ROI Configuration Page (roi-config.html)
- Upload video files for ROI setup
- Interactive canvas for marking parking spots
- Mark 4-point polygons for each parking space
- Video playback controls for precise frame selection
- Save/load ROI configuration to/from server
- Export configuration as JSON file

## Usage Instructions

### 1. Starting the System

1. Start the Python backend server:
   ```bash
   cd server
   python server.py
   ```
   The server will run on `http://localhost:8000`

2. Open the client HTML files in a web browser:
   - Main page: `client/index.html`
   - ROI config: `client/roi-config.html`

### 2. Configuring ROI (Regions of Interest)

1. Click "Configure ROI" button on the main page
2. Upload a video file using the file input
3. Use video controls to navigate to the desired frame
4. **Pause the video** before marking points
5. Click 4 points on the canvas to define each parking spot:
   - Click the 4 corners of each parking space in order
   - Each spot will be automatically completed after 4 clicks
   - Green polygons show completed parking spots
   - Red dots show current points being placed
   - Yellow lines show the polygon being drawn
6. Use controls:
   - **Undo Point**: Remove the last clicked point
   - **Reset All**: Clear all parking spots
   - **Download JSON**: Save configuration locally
   - **Save to Server**: Send configuration to backend
7. Click "← Back to Main" to return to the main page

### 3. Running Detection

1. From the main page, optionally upload a video file
2. Click "Start Detection" to begin processing
3. The video stream will show:
   - Detected vehicles (bounding boxes)
   - Parking spot boundaries
   - Occupancy status (occupied/free)
4. Click "Stop Detection" to halt processing
5. Click "Check Status" to see current system status

## API Endpoints

The backend provides these endpoints:

- `POST /upload-video` - Upload video file
- `POST /start-detection` - Start detection process
- `POST /stop-detection` - Stop detection process
- `GET /status` - Get current system status
- `GET /video-stream` - Video stream with annotations
- `POST /update-roi` - Update ROI configuration
- `GET /get-roi` - Get current ROI configuration

## Technical Details

### Frontend
- Pure HTML/CSS/JavaScript (no frameworks required)
- Canvas-based video annotation interface
- Real-time video streaming via img tag
- AJAX communication with backend API

### Backend
- FastAPI Python web server
- YOLO-based vehicle detection
- OpenCV for video processing
- JSON-based configuration storage
- MJPEG video streaming

## Requirements

### Python Dependencies
- fastapi
- uvicorn
- opencv-python (cv2)
- ultralytics (YOLO)
- numpy
- pydantic

### Browser Requirements
- Modern web browser with HTML5 support
- JavaScript enabled
- Canvas API support

## Troubleshooting

1. **Video not loading**: Check file format compatibility and browser support
2. **Detection not starting**: Ensure video file exists and model is loaded
3. **ROI not saving**: Check network connection and server status
4. **Video stream errors**: Verify detection is running and try refreshing

## Notes

- The system currently uses a hardcoded video path in the server
- ROI configuration is saved to `parking_spots.json`
- Video files should be placed in the `server/video/` directory
- The interface supports common video formats (MP4, MKV, etc.)

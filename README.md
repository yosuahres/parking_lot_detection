# Parking Lot Detection System 🅿️

A real-time parking lot detection system powered by YOLO (You Only Look Once) deep learning model. This web-based application provides automated parking space monitoring with customizable Region of Interest (ROI) configuration capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yosuahres/parking_lot_withcar_detection.git
   cd parking_lot_withcar_detection
   ```

2. **Install Python dependencies**:
   ```bash
   pip install fastapi uvicorn opencv-python ultralytics numpy pydantic python-multipart
   ```

3. **Start the backend server**:
   ```bash
   cd interface/server
   python server.py
   ```
   The server will run on `http://localhost:8000`

4. **Open the web interface**:
   - Main dashboard: Open `interface/client/index.html` in your browser
   - ROI configuration: Open `interface/client/roi-config.html` in your browser

### Usage Instructions

#### 1. Configure Parking Regions (ROI Setup)

1. Navigate to the ROI configuration page
2. Upload a video file using the file input
3. Use video controls to find the perfect frame showing all parking spaces
4. **Pause the video** before marking regions
5. Click 4 points on the canvas to define each parking spot:
   - Click the corners of each parking space in clockwise order
   - Green polygons indicate completed parking spots
   - Red dots show points being placed
   - Yellow lines show the polygon being drawn
6. Management controls:
   - **Undo Point**: Remove the last clicked point
   - **Reset All**: Clear all parking spots
   - **Download JSON**: Save configuration locally
   - **Save to Server**: Upload configuration to backend
7. Return to main page when configuration is complete

#### 2. Run Real-time Detection

1. From the main dashboard, optionally upload a video file
2. Click "Start Detection" to begin processing
3. Monitor the live video stream showing:
   - **White bounding boxes**: Detected vehicles
   - **Green outlines**: Free parking spots
   - **Red outlines**: Occupied parking spots
   - **Spot numbers**: Parking space identifiers
4. Use "Stop Detection" to halt processing
5. Check system status for diagnostics

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | Health check | System status |
| `POST` | `/start-detection` | Start video detection | Success/error message |
| `POST` | `/stop-detection` | Stop video detection | Success/error message |
| `GET` | `/status` | Get system status | Detection status, model info |
| `GET` | `/video-stream` | Live MJPEG video stream | Video stream |

### ROI Configuration Endpoints

| Method | Endpoint | Description | Payload |
|--------|----------|-------------|---------|
| `POST` | `/upload-roi` | Upload ROI JSON file | Multipart form data |
| `POST` | `/update-roi` | Update ROI configuration | JSON configuration |
| `GET` | `/get-roi` | Get current ROI config | None |

### Example API Usage

```javascript
// Start detection
const response = await fetch('http://localhost:8000/start-detection', {
    method: 'POST'
});

// Get system status
const status = await fetch('http://localhost:8000/status');
const data = await status.json();
console.log(data);

// Update ROI configuration
const roiConfig = {
    parking_spots: [
        {
            id: 1,
            points: [[100, 100], [200, 100], [200, 200], [100, 200]]
        }
    ]
};
await fetch('http://localhost:8000/update-roi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(roiConfig)
});
```

---
## 📝 Notes

- The system uses a hardcoded video path (`vid01.mp4`) for demonstration
- ROI configuration is persistently stored in `parking_spots.json`
- Video files should be placed in the `interface/server/video/` directory
- The interface supports common video formats (MP4, MKV, AVI, etc.)
- For production deployment, consider implementing proper authentication and HTTPS


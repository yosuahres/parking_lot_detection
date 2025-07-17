import cv2
import numpy as np
from ultralytics import YOLO
import json

# Load Model
model = YOLO('model_with_83_image.pt')

# Set default values
draw_car_bounding_box = True 
draw_dot_on_car = True 

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        # Get values from JSON; if a key is missing, use the default value
        draw_car_bounding_box = config.get('draw_car_box', True) 
        draw_dot_on_car = config.get('draw_dot_car', True) 
except FileNotFoundError:
    print("Info: 'config.json' not found. Using default settings.")
except json.JSONDecodeError:
    print("Error: 'config.json' is not valid. Using default settings.")


# Try to load the predefined parking spot coordinates from a JSON file
try:
    with open('parking_spots.json', 'r') as f:
        parking_slots = json.load(f)
except FileNotFoundError:
    print("Error: 'parking_slots.json' file not found.")
    exit()

# Convert the loaded parking slot points into NumPy arrays for OpenCV
parking_areas = []
for slot in parking_slots:
    points = np.array(slot['points'], np.int32)
    parking_areas.append(points)

# Processing the video
VIDEO_SOURCE = 'videos/vid01.mkv'
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at '{VIDEO_SOURCE}'")
else:
    # Main loop to process each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing the video or encountered an error.")
            break

        occupancy_status = [False] * len(parking_areas)
        results = model(frame)

        # Process each detected object
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_name = model.names[int(box.cls[0])]
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                point_to_test = (float(center_x), float(center_y))

                # Conditionally draw the dot in the center of the car
                if draw_dot_on_car:
                    dot_color = (42, 42, 165) # Brown color in BGR
                    cv2.circle(frame, (center_x, center_y), 5, dot_color, -1)

                for i, area in enumerate(parking_areas):
                    if cv2.pointPolygonTest(area, point_to_test, False) >= 0:
                        occupancy_status[i] = True
                        break

                # Conditionally draw the bounding box
                if draw_car_bounding_box:
                    box_color = (255, 255, 255)  # White
                    box_thickness = 2
                    label = f"{class_name} {confidence:.2f}"
                    font_color = (0, 0, 0)  # Black
                    font_scale = 0.5
                    font_thickness = 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1), box_color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, font_color, font_thickness)
               

        # Draw occupancy status for each parking spot
        for i, area in enumerate(parking_areas):
            if occupancy_status[i]:
                color = (0, 0, 255)  # Red: Occupied
            else:
                color = (0, 255, 0)  # Green: Empty
            cv2.polylines(frame, [area], isClosed=True,
                          color=color, thickness=3)

        cv2.imshow("Parking Status Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
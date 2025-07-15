import cv2
import json
import numpy as np
from ultralytics import YOLO

# --- FUNGSI UNTUK MEMUAT KOORDINAT DARI JSON ---


def load_parking_spots(file_path):
    """Memuat koordinat spot parkir dari file JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    spots = [np.array(spot['points'], dtype=np.int32) for spot in data]
    return spots


# --- PENGATURAN AWAL ---
model = YOLO('yolov8n.pt')
VIDEO_SOURCE = 'vid01.mkv'
SPOTS_FILE = 'parking_spots.json'

try:
    parking_spots = load_parking_spots(SPOTS_FILE)
except FileNotFoundError:
    print(f"Error: File '{SPOTS_FILE}' tidak ditemukan.")
    exit()

cap = cv2.VideoCapture(VIDEO_SOURCE)

# --- LOOP UTAMA ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau error.")
        break

    results = model(frame, classes=2, conf=0.5)

    # --- DAPATKAN TITIK TENGAH UNTUK SETIAP MOBIL ---
    car_points = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # --- PERBAIKAN DI SINI ---
            # Hitung titik tengah dan pastikan tipenya adalah int standar
            center_x = int((x1 + x2) // 2)
            center_y = int((y1 + y2) // 2)
            car_points.append((center_x, center_y))

    # --- CEK STATUS SETIAP SLOT PARKIR DAN GAMBAR ---
    occupied_count = 0
    for spot_polygon in parking_spots:
        is_occupied = False
        for point in car_points:
            # Sekarang 'point' memiliki tipe data yang benar
            if cv2.pointPolygonTest(spot_polygon, point, False) >= 0:
                is_occupied = True
                break

        if is_occupied:
            color = (0, 0, 255)  # Merah jika terisi
            occupied_count += 1
        else:
            color = (0, 255, 0)  # Hijau jika kosong

        cv2.polylines(frame, [spot_polygon],
                      isClosed=True, color=color, thickness=2)

    brown_color = (19, 69, 139)
    # Loop melalui setiap titik mobil yang sudah dideteksi
    for point in car_points:
        # Gambar lingkaran kecil (titik) pada posisi mobil
        # -1 untuk mengisi lingkaran
        cv2.circle(frame, point, 5, brown_color, -1)

    # --- TAMPILKAN INFORMASI JUMLAH SLOT ---
    total_spots = len(parking_spots)
    available_spots = total_spots - occupied_count
    text = f"Tersedia: {available_spots}/{total_spots}"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Deteksi Status Parkir - Tekan 'q' untuk Keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

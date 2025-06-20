import cv2
import pose_media as pm

pose = pm.mediapipe_pose()  # Inisialisasi pose Mediapipe
pt = pose.mp_holistic.Holistic()  # Gunakan holistic untuk mendeteksi pose

# Inisialisasi VideoCapture untuk webcam (biasanya indeks 0)
cap = cv2.VideoCapture(0)

# Periksa apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()
        
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame tidak dapat diakses.")
        break
    
    try:
        frame, results = pose.mediapipe_detection(frame, pt)
    except Exception as e:
        print(f"Error saat deteksi mediapipe: {e}")
        continue

    pose.draw_styled_landmarks(frame, results)
    resizeImage = cv2.resize(frame, (640, 480))

    cv2.imshow('MediaPipe Pose & Hand Landmarks - Tekan q untuk keluar', resizeImage)

    # Cek apakah tombol 'q' ditekan untuk keluar
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Keluar dari program...")
        break
    
# Setelah loop selesai, lepaskan VideoCapture dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
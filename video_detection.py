import pose_media as pm
import numpy as np
import tensorflow as tf
import cv2
import time

threshold = 0.5  # Threshold untuk menentukan aksi
pTime = 0
cTime = 0
actions = np.array(["DRINKPIL", "NOACT"])  # Label aksi
pose = pm.mediapipe_pose()  # Inisialisasi pose Mediapipe
pt = pose.mp_holistic.Holistic()  # Gunakan holistic untuk mendeteksi pose
new_model = tf.keras.models.load_model('weight\minum_obat_fix.h5')  # Load model yang sudah dilatih
counter = 0  # Inisialisasi counter untuk menghitung aktivitas "minum obat"

sequence = []
sentence = []

video_path="data_test\Jhonatur_28 Juni 2025.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()
    
while cap.isOpened():
    ret, frame = cap.read()
    # rotate = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) -- Untuk video yang terbalik
    flip = cv2.flip(frame, 1)
    if not ret:
        print("Error: Frame tidak dapat diakses.")
        break
    
    try:
        frame, results = pose.mediapipe_detection(flip, pt)
    except Exception as e:
        print(f"Error saat deteksi mediapipe: {e}")
        continue

    pose.draw_styled_landmarks(flip, results)
    keypoints = pose.extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  

    if len(sequence) == 30:
        res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                    if actions[np.argmax(res)] == "DRINKPIL":
                        counter += 1  
            else:
                sentence.append(actions[np.argmax(res)])
                if actions[np.argmax(res)] == "DRINKPIL":
                    counter += 1 
        if len(sentence) > 1: 
            sentence = sentence[-1:]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(flip, str(sentence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(flip, "Counter: " + str(counter), (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    resizeImage = cv2.resize(flip, (640, 480))

    cv2.imshow('Detect Action', resizeImage)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
file = input("name : ")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Use 'MP4V' codec for better compatibility
out = cv2.VideoWriter('test/drinkpil/{}.mp4'.format(file), fourcc, 20.0, (640, 480))

# Record for 30 seconds
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
        cv2.imshow('Video', frame)

        # Check if 15 seconds have elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time > 11:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

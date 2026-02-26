import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ==== Load HandLandmarker Model ====
model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# ==== Camera ====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
INDEX_TIP = 8
prev_points = {}

# ==== Colors & Brush Sizes ====
palette = [(0,255,0),(255,0,0),(0,0,255),(0,255,255),(255,0,255)]  # Green, Blue, Red, Yellow, Purple
draw_color = palette[0]

brush_sizes = [4,8,12,20]
brush_size = 8

# ==== UI Positions ====
color_buttons = [(1200, 150), (1200, 250), (1200, 350), (1200, 450), (1200, 550)]
size_buttons = [(1100, 200), (1100, 300), (1100, 400), (1100, 500)]
clear_zone = (1000, 600, 1180, 670)

# ==== Smooth finger movement ====
def smooth_move(prev, current, alpha=0.3):
    if prev == (0,0):
        return current
    x = int(prev[0] + alpha * (current[0] - prev[0]))
    y = int(prev[1] + alpha * (current[1] - prev[1]))
    return (x,y)

frame_timestamp = 0
prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape

    if canvas is None:
        canvas = np.zeros((h,w,3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    clear_triggered = False

    # ==== Hand Detection & Drawing ====
    if results.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            index_x, index_y = lm_list[INDEX_TIP]

            # ==== Clear Button Trigger ====
            x1,y1,x2,y2 = clear_zone
            if x1 < index_x < x2 and y1 < index_y < y2:
                clear_triggered = True

            # ==== Color Selection ====
            for i, (cx,cy) in enumerate(color_buttons):
                if np.hypot(index_x - cx, index_y - cy) < 30:
                    draw_color = palette[i]

            # ==== Brush Size Selection ====
            for i, (cx,cy) in enumerate(size_buttons):
                if np.hypot(index_x - cx, index_y - cy) < 20:
                    brush_size = brush_sizes[i]

            # ==== Drawing Logic ====
            fingers_up = lm_list[8][1] < lm_list[6][1]  # Index finger up
            if fingers_up:
                prev = prev_points.get(hand_idx,(0,0))
                smoothed = smooth_move(prev,(index_x,index_y))
                if prev != (0,0):
                    cv2.line(canvas, prev, smoothed, draw_color, brush_size)
                prev_points[hand_idx] = smoothed
            else:
                prev_points[hand_idx] = (0,0)

            # ==== Draw Hand Landmarks ====
            for point in lm_list:
                cv2.circle(frame, point, 4, (0,255,0), -1)

    # ==== Execute Clear ====
    if clear_triggered:
        canvas[:] = 0

    # ==== Overlay Canvas ====
    frame = cv2.addWeighted(frame,1,canvas,1,0)

    
    # ---- Color Buttons with halo ----
    for i, (cx,cy) in enumerate(color_buttons):
        color = palette[i]
        cv2.circle(frame,(cx,cy),36,(255,255,255),2)  # halo
        cv2.circle(frame,(cx,cy),30,color,-1)
        if draw_color == color:
            cv2.circle(frame,(cx,cy),40,(255,255,255),3)

    # ---- Brush Size Buttons ----
    for i,(cx,cy) in enumerate(size_buttons):
        size = brush_sizes[i]
        cv2.circle(frame,(cx,cy),size+4,(150,150,150),2)  # outline
        cv2.circle(frame,(cx,cy),size,(200,200,200),-1)
        if brush_size == size:
            cv2.circle(frame,(cx,cy),size+8,(255,255,255),2)
        cv2.putText(frame,f"{size}px",(cx-40,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    # ---- Clear Button ----
    x1,y1,x2,y2 = clear_zone
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),-1)
    cv2.putText(frame,"CLEAR",(x1+15,y1+45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)

    # ==== FPS Counter ====
    curr_time = time.time()
    fps = 1/(curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame,f'FPS: {int(fps)}',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Air Whiteboard Pro - Enhanced UI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

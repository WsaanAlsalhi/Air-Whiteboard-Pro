import cv2
import mediapipe as mp
import numpy as np

# ---- Mediapipe Setup ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,               # Detect up to 2 hands
    min_detection_confidence=0.6,  # Minimum confidence for detection
    min_tracking_confidence=0.7    # Minimum confidence for tracking
)
mp_draw = mp.solutions.drawing_utils  # Used to draw hand landmarks

# ---- Camera Setup ----
cap = cv2.VideoCapture(0)  # Open webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None  # Will hold the drawing
INDEX_TIP = 8  # Index finger tip landmark ID
prev_points = {}  # Stores previous finger positions

# ---- Color Palette ----
palette = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255)   # Purple
]
draw_color = palette[0]  # Default drawing color (green)

# ---- Brush Sizes ----
brush_sizes = [4, 8, 12, 20]
brush_size = 8

# ---- Button Positions (on the right side of the screen) ----
color_buttons = [(1200, 150), (1200, 250), (1200, 350), (1200, 450), (1200, 550)]
size_buttons = [(1100, 200), (1100, 300), (1100, 400), (1100, 500)]
clear_zone = (1000, 600, 1180, 670)  # Area for "Clear" button

# ---- Smooth finger movement to avoid jitter ----
def smooth_move(prev, current, alpha=0.3):
    if prev == (0, 0):
        return current
    x = int(prev[0] + alpha * (current[0] - prev[0]))
    y = int(prev[1] + alpha * (current[1] - prev[1]))
    return (x, y)

# ---- Main Loop ----
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)  # Create a black canvas

    # Convert to RGB for Mediapipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    clear_triggered = False

    # ---- Hand Detection ----
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Convert hand landmarks to pixel coordinates
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            index_x, index_y = lm_list[INDEX_TIP]

            # ---- Clear canvas if finger enters the clear zone ----
            (x1, y1, x2, y2) = clear_zone
            if x1 < index_x < x2 and y1 < index_y < y2:
                clear_triggered = True

            # ---- Color selection ----
            for i, (cx, cy) in enumerate(color_buttons):
                if np.hypot(index_x - cx, index_y - cy) < 30:
                    draw_color = palette[i]

            # ---- Brush size selection ----
            for i, (cx, cy) in enumerate(size_buttons):
                if np.hypot(index_x - cx, index_y - cy) < 20:
                    brush_size = brush_sizes[i]

            # ---- Drawing logic ----
            # Count raised fingers to check if only one (index) finger is up
            fingers_up = sum([
                lm_list[8][1] < lm_list[6][1],   # Index finger
                lm_list[12][1] < lm_list[10][1], # Middle finger
                lm_list[16][1] < lm_list[14][1], # Ring finger
                lm_list[20][1] < lm_list[18][1], # Pinky
                lm_list[4][0] < lm_list[3][0]    # Thumb
            ])

            # If only one finger is up → draw
            if fingers_up == 1:
                prev_x, prev_y = prev_points.get(hand_idx, (0, 0))
                smoothed = smooth_move((prev_x, prev_y), (index_x, index_y))
                if prev_x == 0 and prev_y == 0:
                    prev_points[hand_idx] = smoothed
                else:
                    cv2.line(canvas, (prev_x, prev_y), smoothed, draw_color, brush_size)
                    prev_points[hand_idx] = smoothed
            else:
                prev_points[hand_idx] = (0, 0)

            # Draw Mediapipe hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---- Execute actions ----
    if clear_triggered:
        canvas[:] = 0

    # ---- Combine drawing with live video ----
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # ---- User Interface (buttons) ----

    # Color buttons
    for i, (cx, cy) in enumerate(color_buttons):
        color = palette[i]
        cv2.circle(frame, (cx, cy), 30, color, -1)
        if draw_color == color:
            cv2.circle(frame, (cx, cy), 36, (255, 255, 255), 3)

    # Brush size buttons
    for i, (cx, cy) in enumerate(size_buttons):
        size = brush_sizes[i]
        cv2.circle(frame, (cx, cy), size, (200, 200, 200), -1)
        if brush_size == size:
            cv2.circle(frame, (cx, cy), size + 8, (255, 255, 255), 2)
        cv2.putText(frame, f"{size}px", (cx - 40, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Clear button
    (x1, y1, x2, y2) = clear_zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.putText(frame, "CLEAR", (x1 + 15, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # ---- Display the result ----
    cv2.imshow("Air Whiteboard Pro", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

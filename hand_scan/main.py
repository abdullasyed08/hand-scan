import cv2
import mediapipe as mp
import numpy as np

# -------------------------------------------------------
# Hand Scanner Project - Written by Abdulla
# -------------------------------------------------------

# Load template hand image (with alpha channel if available)
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)

# Resize template to a standard size (adjustable)
template = cv2.resize(template, (800, 800))  
template_h, template_w = template.shape[:2]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -------------------------------------------------------
# Function: Overlay transparent image on frame
# -------------------------------------------------------
def overlay_transparent(background, overlay, position):
    x, y = position
    alpha_overlay = overlay[:, :, 3] / 255.0  # Transparency from PNG
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):  # Loop through color channels
        background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c]
        )

# -------------------------------------------------------
# Main Loop
# -------------------------------------------------------
video_played = False  # flag to avoid continuous video playback

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror effect for natural hand movement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Position template in the center of the screen
    center_x = w // 2 - template_w // 2
    center_y = h // 2 - template_h // 2

    # Overlay hand template on frame
    overlay_transparent(frame, template, (center_x, center_y))

    # Hand detection with Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on detected hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box of hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            # Check if hand is inside template box
            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):

                # If hand fits template -> Play video once
                if not video_played:
                    video = cv2.VideoCapture('vid.mp4')
                    while video.isOpened():
                        ret_vid, frame_vid = video.read()
                        if not ret_vid:
                            break
                        cv2.imshow('Hand Scanner', frame_vid)
                        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit video
                            break
                    video.release()
                    video_played = True  # mark video as played
    else:
        # Reset when no hand detected -> allows replay
        video_played = False  

    # Show the final frame
    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

# -------------------------------------------------------
# Cleanup
# -------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
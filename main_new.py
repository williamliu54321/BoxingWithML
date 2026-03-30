"""
BoxingWithML - Ported to new MediaPipe Tasks API
Original algorithm by erfansn
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from scipy.stats import linregress

# Download hand landmarker model if needed
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# Setup hand landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# State variables (same as original)
hands_area = [[], []]
counter = 0
leftHand_positions = [np.array([0, 0]), np.array([0, 0])]
rightHand_positions = [np.array([0, 0]), np.array([0, 0])]
damping_factory = 0.15

cap = cv2.VideoCapture(0)

print("Starting BoxingWithML punch counter...")
print("Press 'q' to quit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Mirror
    h, w, c = img.shape

    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    # Detect hands
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.hand_landmarks):
            x_list, y_list = [], []

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            top_left_point_hand = np.array([x_min, y_min])
            bottom_right_point_hand = np.array([x_max, y_max])

            # Get handedness
            handedness = results.handedness[hand_id][0].category_name

            if handedness == "Left":
                current_top_left = leftHand_positions[0] + (top_left_point_hand - leftHand_positions[0]) * damping_factory
                current_bottom_right = leftHand_positions[1] + (bottom_right_point_hand - leftHand_positions[1]) * damping_factory
                handedness_id = 0
                hand_positions = leftHand_positions
            else:
                current_top_left = rightHand_positions[0] + (top_left_point_hand - rightHand_positions[0]) * damping_factory
                current_bottom_right = rightHand_positions[1] + (bottom_right_point_hand - rightHand_positions[1]) * damping_factory
                handedness_id = 1
                hand_positions = rightHand_positions

            max_width = current_bottom_right[0] - current_top_left[0]
            max_height = current_bottom_right[1] - current_top_left[1]
            hands_area[handedness_id].append(max_width * max_height)

            # Draw bounding box
            cv2.rectangle(
                img,
                (int(current_top_left[0]), int(current_top_left[1])),
                (int(current_bottom_right[0]), int(current_bottom_right[1])),
                (255, 255, 255), 2
            )

            hand_positions[0] = current_top_left
            hand_positions[1] = current_bottom_right

    # Check for punches (same algorithm as original)
    if len(hands_area[0]) >= 5 and len(hands_area[1]) >= 5:
        left = hands_area[0][-5:]
        right = hands_area[1][-5:]

        x = list(range(5))

        left_hand_slope, _, _, _, _ = linregress(x, left)
        right_hand_slope, _, _, _, _ = linregress(x, right)

        if left_hand_slope > 600:
            print(f"Left Punch {counter}: {left_hand_slope:.0f}")
            counter += 1
        elif right_hand_slope > 600:
            print(f"Right Punch {counter}: {right_hand_slope:.0f}")
            counter += 1

        hands_area[0] = []
        hands_area[1] = []

    # Display counter
    cv2.putText(img, f"Punches: {counter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("BoxingWithML", img)

    if cv2.waitKey(8) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

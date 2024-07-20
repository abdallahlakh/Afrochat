import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import json  # Import the json module

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    32: 'A', 33: 'B', 34: 'L', 0: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'J', 8: 'K', 9: 'M', 
    10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 19: 'W', 20: 'X', 21: 'Y', 22: 'Z', 
    23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9'
}

last_detected_character = None  # Initialize variable to store the last detected character
detected_characters=[]
while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                z_.append(landmark.z)

            center_x, center_y, center_z = np.mean(x_), np.mean(y_), np.mean(z_)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                data_aux.extend([x - min(x_), y - min(y_), z - min(z_), distance_from_center])

    if cv2.waitKey(1) & 0xFF == 32:  # ASCII code for space key is 32
        prediction = model.predict([np.asarray(data_aux)])
        last_detected_character = labels_dict[int(prediction[0])]
        print(f"Detected Character: {last_detected_character}")
        detected_characters.append(last_detected_character)
        with open('detected_characters.json', 'w') as json_file:
           json.dump({"last_detected_character": detected_characters}, json_file)
    with open('detected_characters.json', 'w') as json_file:
        json.dump({"last_detected_character": detected_characters}, json_file)


    if last_detected_character and x_ and y_:
        # Display the last detected character
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        cv2.putText(frame, last_detected_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
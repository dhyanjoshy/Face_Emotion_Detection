import cv2
from keras.models import model_from_json
import numpy as np
import pygetwindow as gw
import pyautogui

# Load the pre-trained model
json_file = open("Face_Emotion_Recognition_Machine_Learning-main\emotiondetector2.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Face_Emotion_Recognition_Machine_Learning-main\emotiondetector2.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def capture_window(window_title):
    try:
        # Get the window object using its title
        window = gw.getWindowsWithTitle(window_title)[0]
        # Get the window position and size
        left, top, width, height = window.left, window.top, window.width, window.height
        # Capture the screen
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        # Convert the screenshot to an OpenCV image
        img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return img_bgr
    except IndexError:
        print("Window not found.")
        return None

labels = {0 : 'angry', 1 : 'happy', 2 : 'neutral', 3 : 'sad', 4 : 'surprise'}

# Window title of the window you want to capture
window_title = "WhatsApp"

while True:
    captured_img = capture_window(window_title)
    if captured_img is not None:
        gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(captured_img, 1.3, 5)
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(captured_img, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            print("Predicted Output:", prediction_label)
            cv2.putText(captured_img, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", captured_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

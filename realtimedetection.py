import cv2
from keras.models import model_from_json
import numpy as np

# Load model
with open("models/emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("models/emotiondetector.weights.h5")

# Load Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Labels
labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

# Open webcam
webcam = cv2.VideoCapture(2)
if not webcam.isOpened():
    print("Cannot access camera")
    exit()

while True:
    ret, im = webcam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # use gray image

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255,0,0), 2)
        face_img = cv2.resize(face_img, (48,48))
        img = extract_features(face_img)
        pred = model.predict(img)
        emotion = labels[pred.argmax()]
        cv2.putText(im, emotion, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

    cv2.imshow("Emotion Recognition", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
        break

webcam.release()
cv2.destroyAllWindows()

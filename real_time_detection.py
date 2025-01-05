import cv2  
import cv2.data  
import tensorflow as tf  
from keras_preprocessing.image import load_img  
import numpy as np  

# Load the emotion detection model  
model = tf.keras.models.load_model('emotion_models.keras')  

# Load the face cascade  
file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  
face_cascade = cv2.CascadeClassifier(file)  

def extract_feature(image):  
    image = cv2.resize(image, (48, 48))  
    feature = np.array(image, dtype=np.float32)  
    feature = feature / 255.0  
    feature = feature.reshape(1, 48, 48, 1)  
    return feature  

# Initialize webcam  
webcam = cv2.VideoCapture(0)  

# Check if the webcam opened successfully  
if not webcam.isOpened():  
    print("Error: Could not open webcam.")  
    exit()  # Exit if the webcam could not be opened  

# Emotion labels  
label = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}  

while True:  
    ret, im = webcam.read()  
    if not ret:  
        print("Failed to capture image")  
        break  
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
    
    if face_cascade.empty():  
        print("Error loading cascade classifier")  
    
    try:  
        for (x, y, w, h) in faces:  
            image = gray[y:y+h, x:x+w]  
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)  
            img = extract_feature(image)  
            pred = model.predict(img)  
            prediction_label = label[pred.argmax()]  
            cv2.putText(im, prediction_label, (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)  

        cv2.imshow('Output', im)  
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit  
            break  
    except cv2.error as e:  
        print(f"OpenCV error: e'")  
        break  

# Release resources  
webcam.release()  
cv2.destroyAllWindows()
import cv2
from keras.models import model_from_json
import numpy as np
import pywhatkit as kit
from datetime import datetime
import time
import yaml


with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

phone_number = config_data.get("phone_number", "")
paciente = config_data.get("paciente", "")
message = None

# from keras_preprocessing.image import load_img
json_file = open("emotion_detector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotion_detector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0


webcam = cv2.VideoCapture(0)
# labels = {0: 'angry', 1: 'disgust', 2: 'fear',
#           3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
labels = {0: 'Nervoso(a)', 1: 'Nojo', 2: 'Medo',
          3: 'Feliz', 4: 'Neutro', 5: 'Bravo(a)', 6: 'Surpreso(a)'}

while True:

    agora = datetime.now()
    hora_atual = agora.hour
    minuto_atual = agora.minute + 1

    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            if prediction_label == 0:
                # Specify the phone number including the country code and the message
                message = f"Olá, o paciente {paciente}, está nervoso!"
            elif prediction_label == 2:
                # Specify the phone number including the country code and the message
                message = f"Olá, o paciente {paciente}, está com medo!"
            elif prediction_label == 6:
                # Specify the phone number including the country code and the message
                message = f"Olá, o paciente {paciente}, está surpreso!"
            elif prediction_label == 3:
                # Specify the phone number including the country code and the message
                message = f"Olá, o paciente {paciente}, está feliz!"
            else:
                message = f"Olá, o paciente {paciente}, está {prediction_label}!"

            kit.sendwhatmsg(phone_number, message, hora_atual, minuto_atual, tab_close=True, close_time=2)
            time.sleep(3)

            # cv2.putText(im, prediction_label)
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass

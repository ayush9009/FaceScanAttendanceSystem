import os
import numpy as np
import pandas as pd
import cv2 as cv
from datetime import datetime

if os.path.exists('id-names.csv'):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
else:
    id_names = pd.DataFrame(columns=['id', 'name'])
    id_names.to_csv('id-names.csv')

if not os.path.exists('faces'):
    os.makedirs('faces')

print('Welcome!')
print('\nPlease enter your ID : ')
print('If this is your first time choose a random ID between 1-1000')

id = int(input('ID: '))
name = ''

if id in id_names['id'].values:
    name = id_names[id_names['id'] == id]['name'].item()
    print(f'Welcome Back {name}!!')
else:
    name = input('Please Enter you name: ')
    os.makedirs(f'faces/{id}')
    id_names = id_names._append({'id': id, 'name': name}, ignore_index=True)
    id_names.to_csv('id-names.csv')

print("\nLet's capture!")

print("Press the 's' key to capture a picture", end=" ")
print("It is recommended to take atleast 30-35 pictures") 
input("\nPress ENTER to start when you're ready, and press the 'q' key to quit when you're done!")

camera = cv.VideoCapture(0)
face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')
photos_taken = 0

while(cv.waitKey(1) & 0xFF != ord('q')):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        xx = x
        x = int(x*1.1)
        xx = int(xx*.9)
        face_region = grey[y:y + h, x:x + w]
        if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
            face_img = cv.resize(face_region, (220, 220))
            img_name = f'face.{id}.{datetime.now().microsecond}.jpg'
            cv.imwrite(f'faces/{id}/{img_name}', face_img)
            photos_taken += 1
            print(f'{photos_taken} -> Photos taken!')

    cv.imshow('Face', img)

camera.release()
cv.destroyAllWindows()
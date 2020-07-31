# librerias

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os






# funcion para detectar y predecir mascara


def detectar_mascara(frame, net_cara, net_mascara):
    # se tomen las dimensiones del frame y se construye un blob desde ahi
    h, w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pasa el blob por la red y se obtiene la deteccion de caras
    net_cara.setInput(blob)
    detecciones=net_cara.forward()

    # listas de caras, localizaciones y predicciones
    caras=[]
    locs=[]
    preds=[]

    # bucle sobre las detecciones
    for i in range(0, detecciones.shape[2]):
        # probabilidad asociada a la deteccion, umbral de confianza
        confianza=detecciones[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confianza>CONFIANZA:
            # coordenadas (x, y) del contorno de la caja del objeto
            caja=detecciones[0, 0, i, 3:7]*np.array([w, h, w, h])
            x_start, y_start, x_end, y_end=caja.astype('int')

            # asegurar que los limites de la caja estan en el frame
            x_start, y_startY=(max(0, x_start), max(0, y_start))
            x_end, y_end=(min(w-1, x_end), min(h-1, y_end))

            # extraer ROI de cara, pasar a RGB, redimensionar a 224x224 y preprocesar
            cara=frame[y_start:y_end, x_start:x_end]
            cara=cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
            cara=cv2.resize(cara, (224, 224))
            cara=img_to_array(cara)
            cara=preprocess_input(cara)

            # añadir a las listas
            caras.append(cara)
            locs.append((x_start, y_start, x_end, y_end))

    # solo se hacen predicciones si hay una cara detectada
    if len(caras)>0:
        # todas las caras a la vez
        caras=np.array(caras, dtype='float32')
        preds=net_mascara.predict(caras, batch_size=32)

        
    return locs, preds









CONFIANZA=.5




# se carga el modelo detector de caras

prototxt=os.path.sep.join(['detector', 'deploy.prototxt'])
pesos=os.path.sep.join(['detector', 'res10_300x300_ssd_iter_140000.caffemodel'])


net_cara=cv2.dnn.readNet(prototxt, pesos)

# se carga el modelo detector de mascarillas
net_mascara=load_model('detector_mascara.model')






# inicializa camara web

cam=VideoStream(src=0).start()
time.sleep(2.0)




# bucle sobre los frames
while 1:
    # coge el frame del video y redimensiona a 400 pixels
    frame=cam.read()
    frame=imutils.resize(frame, width=400)

    # detectar caras en el frame y determinar si hay o no mascara
    locs, preds=detectar_mascara(frame, net_cara, net_mascara)

    # bucle sobre las caras detectadas y sus localizaciones
    for caja, pred in zip(locs, preds):
        x_start, y_start, x_end, y_end=caja
        con_mascara, sin_mascara=pred

        # determinar la etiqueta y color para dibujar caja y texto
        etiqueta='Mascara' if con_mascara>sin_mascara else 'Sin Mascara'
        color=(0, 255, 0) if etiqueta=='Mascara' else (0, 0, 255)

        # incluye la probabilidad en la etiqueta
        etiqueta='{}: {:.2f}%'.format(etiqueta, max(con_mascara, sin_mascara)*100)

        # enseña la etiqueta y el limite de la caja en el frame
        cv2.putText(frame, etiqueta, (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

    # muestra el frame de salida
    cv2.imshow('Camara', frame)
    key=cv2.waitKey(1)&0xFF

    # si se presiona `q`, rompe el bucle
    if key==ord('q'):
        break

# limpia pantall
cv2.destroyAllWindows()
cam.stop()













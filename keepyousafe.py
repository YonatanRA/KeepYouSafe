import cv2
import time
import imutils
import os
import numpy as np

from math import pow, sqrt
from imutils.video import FPS
from imutils.video import VideoStream

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from flask import Flask, render_template, Response


PATH=os.path.dirname(os.path.abspath(__file__))

# iniciaciliza la aplicacion
app=Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')



modelo=''
net_cara=''
net_mascara=''

@app.before_first_request  # antes del primer requests
def startup():
    global modelo, net_cara, net_mascara

    # cargar el modelo caffe
    modelo=cv2.dnn.readNetFromCaffe(PATH+'detector/SSD_MobileNet_prototxt.txt', PATH+'detector/SSD_MobileNet.caffemodel')

    prototxt=os.path.sep.join([PATH+'detector', 'deploy.prototxt'])
    pesos=os.path.sep.join([PATH+'detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

    net_cara=cv2.dnn.readNet(prototxt, pesos)

    # se carga el modelo detector de mascarillas
    net_mascara=load_model(PATH+'detector/detector_mascara.model')




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

        # detecciones mayor que un umbral de confianza
        if confianza>.5:
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



def deteccion():
    cam=VideoStream(src=0).start()
    fps=FPS().start()
    time.sleep(2.0)

    # bucle sobre los frames
    while 1:
        frame=cam.read()
        frame=imutils.resize(frame, width=900)


        ### MASCARA
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



        ###DISTANCIA
        h, w=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), .007843, (300, 300), 127.5)

        modelo.setInput(blob)
        detecciones=modelo.forward()


        F=615   # hiperparametro

        pos={}
        coordenadas={}


        for i in np.arange(0, detecciones.shape[2]):
            confianza=detecciones[0, 0, i, 2]

            if confianza>.5:
                objecto=int(detecciones[0, 0, i, 1])

                if(objecto==15):
                    caja=detecciones[0, 0, i, 3:7]*np.array([w, h, w, h])
                    x_start, y_start, x_end, y_end=caja.astype('int')

                    etiqueta='Persona: {:.2f}%'.format(confianza*100)
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (10,255,0), 2)
                    y=y_start-15 if y_start-15>15 else y_start+15
                    cv2.putText(frame, etiqueta, (x_start, y), cv2.FONT_HERSHEY_DUPLEX, .5, (20,255,0), 1)

                    coordenadas[i]=(x_start, y_start, x_end, y_end)

                    #M punto medio de la caja
                    x_medio=round((x_start+x_end)/2, 4)
                    y_medio=round((y_start+y_end)/2, 4)

                    altura_media=round(y_end-y_start, 4)

                    # distancia desde la camara basada en similitud de triangulos
                    distancia=(F*165)/altura_media

                    # punto medio de la caja de contorno (en cm) basada en similitud de triangulos
                    x_medio_cm=(x_medio*distancia)/F
                    y_medio_cm=(y_medio*distancia)/F

                    pos[i]=(x_medio_cm, y_medio_cm, distancia)

        proximidad=[]

        # bucle por las posiciones
        for i in pos.keys():
            for j in pos.keys():
                if i<j:
                    # se calcula la distancia por Euclides
                    dist=sqrt(pow(pos[i][0]-pos[j][0],2)+pow(pos[i][1]-pos[j][1],2)+pow(pos[i][2]-pos[j][2],2))

                    # distancia umbral-175 cm
                    if dist<175:
                        proximidad.append(i)
                        proximidad.append(j)

                        aviso='Manten la distancia de seguridad.¡Muevete!'
                        cv2.putText(frame, aviso, (50,50), cv2.FONT_HERSHEY_DUPLEX, .5, color, 1)


        for i in pos.keys():
            if i in proximidad:
                color=[0,0,255]
            else:
                color=[0,255,0]

            x, y, w, h=coordenadas[i]

            cv2.rectangle(frame, (x, y), (w, h), color, 2)



        # muestra el frame de salida
        frame=cv2.imencode('.jpg', frame)[1].tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key=cv2.waitKey(1)&0xFF

        # si se presiona `q`, rompe el bucle
        if key==ord('q'):
            break

        fps.update()

    fps.stop()

    # limpia pantalla
    cv2.destroyAllWindows()
    cam.stop()




@app.route('/video')
def video():
    return Response(deteccion(), mimetype='multipart/x-mixed-replace; boundary=frame')



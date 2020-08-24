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








# funcion para calcular la distancia social
def distancia_social():
	cam=VideoStream(src=0).start()
	cam.stop
	cam=VideoStream(src=0).start()
	
	fps=FPS().start()
    
	# bucle en el video
	while 1:
		frame=cam.read()
		frame=imutils.resize(frame, width=600)  # 600 pixels de pantalla
		
		
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
							
		frame=cv2.imencode('.jpg', frame)[1].tobytes()
		
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		
		key=cv2.waitKey(1)&0xFF

		if key==ord('q'):
			break
		fps.update()
		
	fps.stop()
